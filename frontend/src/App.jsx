import { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

const API_URL = import.meta.env.VITE_API_URL || '';
const TICKERS = [
  'TCS.NS',
  'INFY.NS',
  'RELIANCE.NS',
  'HDFCBANK.NS',
  'SBIN.NS',
  'ICICIBANK.NS'
];

const defaultPayload = {
  ticker: 'TCS.NS',
  start_date: '2026-01-01',
  end_date: '2026-05-14',
  window_size: 60,
  model_type: 'random_forest'
};

function formatCurrency(value) {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2
  }).format(value);
}

function App() {
  const [requestData, setRequestData] = useState(defaultPayload);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('Enter a symbol and build a Stocksense forecast.');
  const [result, setResult] = useState(null);

  const chartData = useMemo(() => {
    if (!result?.predictions) return [];
    return result.predictions.map((item) => ({
      date: item.date,
      actual: item.actual,
      predicted: item.predicted
    }));
  }, [result]);

  const handleChange = (field, value) => {
    setRequestData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setMessage('Fetching forecasts from Stocksense...');
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to fetch prediction.');
      }
      setResult(data);
      setMessage(`Stocksense generated predictions for ${data.ticker}.`);
    } catch (error) {
      setMessage(error.message || 'Unexpected server error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header>
        <div className="brand-row">
          <span className="brand-mark">S</span>
          <div>
            <h1>Stocksense</h1>
            <p>AI-powered stock price forecasting with market signals, timelines, and accuracy metrics.</p>
          </div>
        </div>
      </header>

      <main>
        <section className="form-panel">
          <h2>Forecast Setup</h2>
          <form onSubmit={handleSubmit}>
            <label>
              Stock Ticker
              <input
                list="ticker-suggestions"
                value={requestData.ticker}
                onChange={(e) => handleChange('ticker', e.target.value.toUpperCase())}
                placeholder="e.g. TCS.NS"
                required
              />
              <datalist id="ticker-suggestions">
                {TICKERS.map((symbol) => (
                  <option key={symbol} value={symbol} />
                ))}
              </datalist>
            </label>

            <div className="split-row">
              <label>
                Start Date
                <input
                  type="date"
                  value={requestData.start_date}
                  onChange={(e) => handleChange('start_date', e.target.value)}
                  required
                />
              </label>
              <label>
                End Date
                <input
                  type="date"
                  value={requestData.end_date}
                  onChange={(e) => handleChange('end_date', e.target.value)}
                  required
                />
              </label>
            </div>

            <div className="split-row">
              <label>
                Window Size
                <input
                  type="number"
                  min="20"
                  max="120"
                  value={requestData.window_size}
                  onChange={(e) => handleChange('window_size', Number(e.target.value))}
                />
              </label>
              <label>
                Model Type
                <select
                  value={requestData.model_type}
                  onChange={(e) => handleChange('model_type', e.target.value)}
                >
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                </select>
              </label>
            </div>

            <button type="submit" disabled={loading}>
              {loading ? 'Running Stocksense...' : 'Run Forecast'}
            </button>
          </form>

          <div className="alert">{message}</div>
        </section>

        <section className="snapshot-panel">
          <div className="metric-row">
            <div className="metric-card">
              <h3>Selected Ticker</h3>
              <p>{requestData.ticker}</p>
            </div>
            <div className="metric-card">
              <h3>Training Range</h3>
              <p>{requestData.start_date} → {requestData.end_date}</p>
            </div>
            <div className="metric-card">
              <h3>Window</h3>
              <p>{requestData.window_size} days</p>
            </div>
          </div>

          {result && (
            <div className="metric-row">
              <div className="metric-card">
                <h3>Next Forecast</h3>
                <p>{formatCurrency(result.next_prediction.predicted_close)}</p>
              </div>
              <div className="metric-card">
                <h3>Next Trading Day</h3>
                <p>{result.next_prediction.next_date}</p>
              </div>
              <div className="metric-card">
                <h3>Expected Change</h3>
                <p className={result.next_prediction.change_pct >= 0 ? 'positive' : 'negative'}>
                  {result.next_prediction.change_pct.toFixed(2)}%
                </p>
              </div>
            </div>
          )}
        </section>

        {result && (
          <section className="chart-panel">
            <div className="chart-header">
              <div>
                <h2>Actual vs Predicted</h2>
                <p>Performance of Stocksense over the test interval.</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={chartData} margin={{ top: 20, right: 24, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#0f172a', borderRadius: '12px', borderColor: '#334155' }} />
                <Legend wrapperStyle={{ color: '#cbd5e1' }} />
                <Line type="monotone" dataKey="actual" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="predicted" stroke="#f97316" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </section>
        )}

        {result && (
          <section className="metrics-panel">
            <h2>Model Metrics</h2>
            <div className="metric-row">
              <div className="metric-card">
                <h3>MAE</h3>
                <p>{result.metrics.mae.toFixed(2)}</p>
              </div>
              <div className="metric-card">
                <h3>RMSE</h3>
                <p>{result.metrics.rmse.toFixed(2)}</p>
              </div>
              <div className="metric-card">
                <h3>MAPE</h3>
                <p>{result.metrics.mape.toFixed(2)}%</p>
              </div>
              <div className="metric-card">
                <h3>Directional Accuracy</h3>
                <p>{result.metrics.directional_accuracy.toFixed(2)}%</p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
