import os
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

app = Flask(__name__)
client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    ticker   = data.get("ticker", "").strip().upper()
    ohlcv    = data.get("ohlcv", {})
    params   = data.get("params", {})

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    has_ohlcv = all(ohlcv.get(k) for k in ["open", "high", "low", "close"])
    ohlcv_text = (
        f"User-provided OHLCV — Open: {ohlcv['open']}, High: {ohlcv['high']}, "
        f"Low: {ohlcv['low']}, Close: {ohlcv['close']}, Volume: {ohlcv.get('volume', 'N/A')}"
        if has_ohlcv
        else f"No OHLCV data provided — use realistic market estimates for {ticker}."
    )

    prompt = f"""You are an expert quantitative analyst and ML engineer specialising in LSTM-based stock price prediction.

Analyse "{ticker}" with these parameters:
{ohlcv_text}
Model: {params.get('layers', 2)}-layer LSTM, window={params.get('window', 60)} days, dropout={params.get('dropout', 20)}%, horizon={params.get('horizon', '7 Days')}
Market sentiment: {params.get('sentiment', 'Neutral')}

Reply with ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "ticker": "{ticker}",
  "currentPrice": <realistic number>,
  "signal": "BUY" | "HOLD" | "SELL",
  "confidence": <50-95>,
  "predictions": {{"1d": <price>, "7d": <price>, "30d": <price>, "90d": <price>}},
  "metrics": {{"mae": <0.01-5.0>, "rmse": <0.01-8.0>, "directionalAccuracy": <55-90>, "sharpeRatio": <-1.0 to 2.5>}},
  "technicalIndicators": {{
    "rsi": <20-80>, "rsiSignal": "overbought"|"oversold"|"neutral",
    "macd": <number>, "macdSignal": "bullish"|"bearish",
    "bollingerPosition": "upper"|"middle"|"lower",
    "movingAvg50": <price>, "movingAvg200": <price>,
    "ma_cross": "golden_cross"|"death_cross"|"none"
  }},
  "riskScore": <1-10>,
  "volatility": "low"|"moderate"|"high"|"extreme",
  "analysis": {{
    "summary": "<2-3 sentences on current situation and LSTM rationale>",
    "bullCase": "<1-2 sentences>",
    "bearCase": "<1-2 sentences>",
    "lstmInsight": "<1-2 sentences on what the model finds in price sequence patterns>"
  }},
  "priceHistory": [<30 realistic daily close prices, oldest first, ending near currentPrice>]
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
