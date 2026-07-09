import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title='Stock Price Prediction',
    page_icon='📈',
    layout='wide'
)

st.title('Stock Price Prediction Frontend')
st.write('A simple Streamlit app for fetching NSE stock data, training an LSTM model, and visualizing predictions.')

with st.sidebar:
    st.header('Model Configuration')
    ticker = st.text_input('Ticker', value='TCS.NS')
    start_date = st.date_input('Start date', value=pd.to_datetime('2026-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('2026-05-14'))
    window_size = st.slider('Window size', min_value=10, max_value=100, value=60, step=5)
    epochs = st.slider('Epochs', min_value=5, max_value=50, value=15, step=1)
    batch_size = st.selectbox('Batch size', options=[16, 32, 64], index=1)
    st.write('---')
    st.write('Use the buttons below to fetch data and train the model.')
    fetch_data = st.button('Fetch Data')
    train_model_btn = st.button('Train Model')


@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError('No data fetched. Check ticker and date range.')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def create_sequences(features: np.ndarray, targets: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def build_model(window_size: int, feature_count: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, feature_count)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

if fetch_data:
    try:
        st.session_state.raw_data = load_stock_data(ticker, start_date, end_date)
        st.success(f'Data loaded for {ticker} ({len(st.session_state.raw_data)} rows)')
    except Exception as exc:
        st.error(str(exc))

if st.session_state.raw_data is not None:
    df = st.session_state.raw_data
    st.subheader('Recent Data')
    st.dataframe(df.tail(10))

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Start Date', df.index[0].date())
        st.metric('End Date', df.index[-1].date())
    with col2:
        st.metric('Rows', len(df))
        st.metric('Last Close', f'₹{df.Close.iloc[-1]:.2f}')

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df.index, df['Close'], color='steelblue', linewidth=1.3)
    axes[0].set_title('Closing Price')
    axes[0].set_ylabel('Price (INR)')
    axes[0].grid(alpha=0.3)

    axes[1].bar(df.index, df['Volume'], color='darkorange', alpha=0.6)
    axes[1].set_title('Volume')
    axes[1].set_ylabel('Volume')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    if train_model_btn:
        st.subheader('Training the LSTM model')
        with st.spinner('Preprocessing data and training the model...'):
            df_proc = df.copy()
            df_proc.ffill(inplace=True)
            df_proc['next_close'] = df_proc['Close'].shift(-1)
            df_proc.dropna(inplace=True)

            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            target = 'next_close'
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()

            scaled_features = feature_scaler.fit_transform(df_proc[features])
            scaled_target = target_scaler.fit_transform(df_proc[[target]])

            X, y = create_sequences(scaled_features, scaled_target, window_size)
            split_index = int(0.8 * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            model = build_model(window_size, len(features))
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=0
            )

            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
            y_true = target_scaler.inverse_transform(y_test).flatten()

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            dir_true = np.sign(np.diff(y_true))
            dir_pred = np.sign(np.diff(y_pred))
            dir_acc = np.mean(dir_true == dir_pred) * 100

        st.success('Model training complete!')
        st.metric('MAE', f'₹{mae:.2f}')
        st.metric('RMSE', f'₹{rmse:.2f}')
        st.metric('MAPE', f'{mape:.2f}%')
        st.metric('Directional Accuracy', f'{dir_acc:.2f}%')

        dates = df_proc.index[window_size + split_index:window_size + split_index + len(y_true)]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(dates, y_true, label='Actual', color='steelblue', linewidth=1.5)
        ax2.plot(dates, y_pred, label='Predicted', color='darkorange', linestyle='--', linewidth=1.5)
        ax2.set_title('Actual vs Predicted Next-Day Close')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (INR)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        last_window = scaled_features[-window_size:].reshape(1, window_size, len(features))
        next_pred_scaled = model.predict(last_window, verbose=0)
        next_pred_price = target_scaler.inverse_transform(next_pred_scaled)[0, 0]
        last_close = df['Close'].iloc[-1]
        change_pct = (next_pred_price - last_close) / last_close * 100

        st.write('### Next Trading Day Prediction')
        st.write(f'Last available close: **₹{last_close:.2f}** on **{df.index[-1].date()}**')
        st.write(f'Predicted next close: **₹{next_pred_price:.2f}**')
        st.write(f'Expected change: **{change_pct:+.2f}%**')
