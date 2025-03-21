

# 📈 Stock Forecast GUI (Standalone Module)

This is a standalone stock price forecasting GUI built in Python using Tkinter and Matplotlib. It allows users to visualise and compare forecasts from multiple models — LSTM, GRU, ARIMA, Holt-Winters, and Linear Regression — with optional warm-up periods for improved accuracy.

> 🔧 Developed as part of the broader goal to build a fully automated trading pipeline.

---

### 📦 Features

- Toggle between forecast algorithms (LSTM, GRU, etc.)
- Optional warm-up periods for model calibration
- Real-time, timezone-aware plotting
- Auto-fetches historical stock data from Alpaca or Yahoo Finance
- Interactive tooltips using Matplotlib + Mplcursors
- Navigate between multiple tickers in the UI

---

### 🚀 Getting Started

 1. Navigate to the module folder

```
cd PassiveForecaster
```
2. Run the application

```
python main.py
```

Check out the ````Config.py```` file to change things like Selected tickers

🔑 API Key Setup
This app can pull stock data using either:

Yahoo Finance (YF) — a community-driven, public data source
Alpaca — a trading platform with more stable and accurate financial data

To use Alpaca, you must:

1. Create an Alpaca account
2. Generate API keys from your account dashboard
3. Save your keys in a file called AlpacaKeys.py
You can place this file anywhere, and update config.py to load from that location.

Example AlpacaKeys.py:
```
ALPACA_ENDPOINT = "https://paper-api.alpaca.markets/v2"
ALPACA_KEY = "your_alpaca_key"
ALPACA_SECRET = "your_alpaca_secret"
```
⚠️ Do not share your API keys. Keep this file private and secure.


#### 🔁 Switching to Yahoo Finance (YF)
If you don’t have Alpaca keys or prefer not to use them, switch the data source in ````config.py````:

````DATA_SOURCE = "YF" ```` # Options: "Alpaca" or "YF"
⚠️ Note: Yahoo Finance is easier to use but may introduce bias or inconsistencies in pricing data. Use with caution for high-accuracy forecasting.

