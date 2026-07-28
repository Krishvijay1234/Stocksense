from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

app = Flask(__name__)
CORS(app)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT_DIR / 'frontend' / 'dist'

BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TECHNICAL_FEATURES = [
    'return',
    'high_low',
    'open_close',
    'volume_change',
    'sma_5',
    'sma_10',
    'sma_20',
    'ema_12',
    'ema_26',
    'macd',
    'rsi_14',
    'rsi_7'
]
LAG_FEATURES = [
    'close_lag_1',
    'close_lag_2',
    'close_lag_3',
    'close_lag_5',
    'close_lag_10'
]
VOLATILITY_FEATURES = [
    'roll_std_5',
    'roll_std_10',
    'roll_std_20'
]
FEATURE_COLS = BASE_FEATURES + TECHNICAL_FEATURES + LAG_FEATURES + VOLATILITY_FEATURES


@lru_cache(maxsize=64)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError('No data was fetched for the requested ticker or date range.')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[BASE_FEATURES].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['Close'].pct_change().fillna(0)
    df['high_low'] = (df['High'] - df['Low']) / df['Low'].replace(0, np.nan)
    df['open_close'] = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
    df['volume_change'] = df['Volume'].pct_change().fillna(0)

    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()

    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['rsi_7'] = compute_rsi(df['Close'], 7)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def add_lag_and_volatility(df: pd.DataFrame, lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)

    # Rolling volatility (std) features
    for win in (5, 10, 20):
        df[f'roll_std_{win}'] = df['Close'].rolling(window=win).std()

    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def create_sequences(features: np.ndarray, targets: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i].ravel())
        y.append(targets[i])
    return np.array(X), np.array(y)


def build_regressor(model_type: str):
    model_type = model_type.lower()
    if model_type == 'xgboost' and HAS_XGBOOST:
        return XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1, verbosity=0)
    return RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)


def build_secondary_regressor():
    # complementary regressor: prefer XGBoost when available for diversity
    if HAS_XGBOOST:
        return XGBRegressor(n_estimators=120, random_state=7, n_jobs=-1, verbosity=0)
    return RandomForestRegressor(n_estimators=200, random_state=7, n_jobs=-1)


def ensemble_predict(models, X_val, y_val, X_pred):
    """Fit each regressor and compute validation MAE, then weight predictions by inverse MAE.

    models: list of untrained sklearn regressors
    X_val, y_val: validation arrays used to compute model MAE
    X_pred: array to predict (test set or next window)
    Returns: weighted prediction array for X_pred
    """
    preds = []
    maes = []
    trained_models = []

    # small grid search per model for improved validation performance
    for model in models:
        best_m = None
        best_mae = float('inf')
        # simple grid choices depending on estimator type
        if HAS_XGBOOST and isinstance(model, XGBRegressor):
            param_grid = [{'n_estimators': [80, 120], 'max_depth': [3, 6]}]
        else:
            param_grid = [{'n_estimators': [100, 200], 'max_depth': [6, 12, None]}]

        # flatten param grid combos
        combos = []
        for pg in param_grid:
            keys = list(pg.keys())
            from itertools import product
            for vals in product(*(pg[k] for k in keys)):
                combos.append(dict(zip(keys, vals)))

        for combo in combos:
            m = model.__class__(**{k: v for k, v in combo.items() if v is not None})
            try:
                m.fit(X_val['train_X'], X_val['train_y'].to_numpy().ravel())
                p = m.predict(X_val['val_X'])
                mae_val = float(mean_absolute_error(X_val['val_y'].to_numpy().ravel(), p))
                if mae_val < best_mae:
                    best_mae = mae_val
                    best_m = m
            except Exception:
                continue

        if best_m is None:
            # fallback: fit original model
            model.fit(X_val['train_X'], X_val['train_y'].to_numpy().ravel())
            best_m = model
            best_mae = float(mean_absolute_error(X_val['val_y'].to_numpy().ravel(), best_m.predict(X_val['val_X'])))

        trained_models.append(best_m)
        preds.append(best_m.predict(X_pred))
        maes.append(max(best_mae, 1e-6))

    inv = np.array([1.0 / m for m in maes])
    weights = inv / inv.sum()
    stacked = np.vstack(preds)
    weighted = np.dot(weights, stacked)
    return weighted, weights


def build_classifier():
    if HAS_XGBOOST:
        return XGBClassifier(n_estimators=120, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    return RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)


def next_business_date(last_date: pd.Timestamp) -> str:
    candidate = last_date + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.strftime('%Y-%m-%d')


def validate_payload(payload: dict):
    if not isinstance(payload, dict):
        raise ValueError('Payload must be a JSON object.')
    if 'ticker' not in payload:
        raise ValueError('Ticker symbol is required.')
    return {
        'ticker': str(payload.get('ticker', 'TCS.NS')).strip().upper(),
        'start_date': str(payload.get('start_date', '2026-01-01')),
        'end_date': str(payload.get('end_date', '2026-05-14')),
        'window_size': int(payload.get('window_size', 60)),
        'model_type': str(payload.get('model_type', 'random_forest')).lower()
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'app': 'Stocksense Backend'})


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = validate_payload(request.get_json(force=True))
    ticker = payload['ticker']
    start_date = payload['start_date']
    end_date = payload['end_date']
    window_size = payload['window_size']
    model_type = payload['model_type']

    df = load_stock_data(ticker, start_date, end_date)
    df = add_technical_indicators(df)
    df = add_lag_and_volatility(df)
    df['next_close'] = df['Close'].shift(-1)

    if len(df) < window_size + 20:
        raise ValueError('Not enough data to build a dataset for the requested window size.')

    # Preserve the most recent row for the actual next-day forecast
    future_row = df.iloc[[-1]][FEATURE_COLS].copy()
    df = df.iloc[:-1].copy()
    df.dropna(inplace=True)

    X = df[FEATURE_COLS]
    y = df['next_close']

    if len(X) < 50:
        raise ValueError('Not enough historical observations after feature engineering.')

    split_index = int(0.8 * len(X))
    X_train_all, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_all, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_test) == 0:
        raise ValueError('Not enough historical data for the chosen window size and date range.')

    # Further split training into train/validation for ensemble weighting
    val_split = int(0.85 * len(X_train_all))
    train_X, val_X = X_train_all.iloc[:val_split], X_train_all.iloc[val_split:]
    train_y, val_y = y_train_all.iloc[:val_split], y_train_all.iloc[val_split:]

    models = [build_regressor(model_type), build_secondary_regressor()]

    X_val_bundle = {
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y
    }

    y_pred, weights = ensemble_predict(models, X_val_bundle, None, X_test)
    y_true = y_test.values

    # Next day prediction using ensemble and the latest row of live features
    for m in models:
        m.fit(X_train_all, y_train_all.to_numpy().ravel())
    preds_next = np.vstack([m.predict(future_row) for m in models])
    next_pred_price = float(np.dot(weights, preds_next).item())
    last_close = float(future_row['Close'].iloc[0])
    change_pct = float((next_pred_price - last_close) / last_close * 100)

    dates = df.index[split_index:split_index + len(y_true)]
    prediction_points = [
        {
            'date': str(date.date()),
            'actual': float(actual),
            'predicted': float(predicted)
        }
        for date, actual, predicted in zip(dates, y_true, y_pred)
    ]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    directional_accuracy = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)

    return jsonify({
        'ticker': ticker,
        'rows': len(df),
        'model_type': model_type,
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'directional_accuracy': directional_accuracy
        },
        'predictions': prediction_points,
        'next_prediction': {
            'predicted_close': next_pred_price,
            'last_close': last_close,
            'change_pct': change_pct,
            'next_date': next_business_date(df.index[-1])
        }
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404

    if path and (FRONTEND_DIST / path).exists():
        return send_from_directory(FRONTEND_DIST, path)

    if (FRONTEND_DIST / 'index.html').exists():
        return send_from_directory(FRONTEND_DIST, 'index.html')

    return jsonify({'message': 'Frontend build not available. Run the frontend build step first.'}), 404


@app.errorhandler(Exception)
def handle_errors(error):
    message = str(error)
    return jsonify({'error': message}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
