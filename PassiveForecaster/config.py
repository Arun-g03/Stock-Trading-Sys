# config.py

import sys
import os

#DO NOT SHARE KEYS WITH ANYONE
#Put keys in seperate file "AlpacaKeys.py" and import them here
# Add directory containing AlpacaKeys.py to the Python path

# Logging configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Data source
DATA_SOURCE = "Alpaca"  # "YF" or "Alpaca"

# Ticker symbols to track
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL"]

# Forecasting configuration
FORECAST_HORIZON = 10
WARMUP_PERIOD = 24

# Time interval (e.g. 1-minute)
INTERVAL = "1m"

# How far back to look for data per interval
INTERVAL_LOOKBACK_LIMITS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 3650  # ~10 years
}
