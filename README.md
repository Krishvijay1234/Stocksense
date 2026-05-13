# StockSense AI — LSTM Stock Price Prediction

A deployed web app that brings your LSTM stock prediction notebook to life.

## What it does
- Accepts any stock ticker + optional OHLCV data
- Simulates your LSTM pipeline (window size, layers, dropout, horizon)
- Returns price targets (1D / 7D / 30D / 90D), technical indicators, risk score, and AI analysis
- Displays a 30-day price history chart with LSTM forecast overlay

---

## Project Structure
```
stocksense/
├── app.py              # Flask backend + Anthropic API call
├── templates/
│   └── index.html      # Full frontend (single-file, no build step)
├── requirements.txt
├── Procfile            # For Heroku / Render / Railway
└── README.md
```

---

## Local Development

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the server
```bash
python app.py
```
Visit http://localhost:5000

---

## Deploy to Render (recommended — free tier)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Add environment variable:
   - `ANTHROPIC_API_KEY` = your key
6. Click **Deploy**

---

## Deploy to Railway

1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add environment variable: `ANTHROPIC_API_KEY`
4. Railway auto-detects the Procfile — deploy runs automatically

---

## Deploy to Heroku

```bash
heroku create stocksense-ai
heroku config:set ANTHROPIC_API_KEY=sk-ant-...
git push heroku main
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key (get at console.anthropic.com) |
| `PORT` | optional | Port to listen on (default 5000) |

---

## How it maps to your notebook

| Notebook Step | App Equivalent |
|---|---|
| `kagglehub` download + CSV load | User inputs ticker + OHLCV |
| `MinMaxScaler` + sliding window | Sent as params to Claude for simulation |
| LSTM model (64u × 2 layers) | Configurable in sidebar |
| `model.fit` (20 epochs) | AI generates realistic metrics |
| MAE / RMSE / Directional Accuracy | Displayed in metric cards |
| `plt.plot(predicted, actual)` | Interactive Chart.js price chart |
