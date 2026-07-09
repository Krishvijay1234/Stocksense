# Stocksense

Stocksense is an advanced stock price forecasting app with a Flask backend and a React frontend. It streams market history from Yahoo Finance, trains a regression model, and visualizes predicted vs actual stock movement.

## Setup

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install backend dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Backend

```bash
python flask_backend.py
```

The backend listens on `http://localhost:5000` and exposes:

- `GET /api/health`
- `POST /api/predict`

## Run the Frontend

1. Change into the frontend directory:

   ```bash
   cd frontend
   ```

2. Install Node dependencies:

   ```bash
   npm install
   ```

3. Start the React UI:

   ```bash
   npm run dev
   ```

Open the URL shown by Vite (usually `http://localhost:5173`).

## Stocksense Features

- Stock ticker forecasting with configurable history window
- Random Forest or XGBoost model selection
- Actual vs predicted price visualization
- Error metrics: MAE, RMSE, MAPE, directional accuracy
- Next trading day close forecast and expected change

## Notes

- If XGBoost is not installed, Stocksense falls back to Random Forest automatically.
- Use valid NSE ticker symbols like `TCS.NS`, `INFY.NS`, or `RELIANCE.NS`.
