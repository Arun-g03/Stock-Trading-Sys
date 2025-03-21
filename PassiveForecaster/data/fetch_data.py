import logging
import pandas as pd
import yfinance as yf
from config import INTERVAL, INTERVAL_LOOKBACK_LIMITS, DATA_SOURCE, ALPACA_KEY, ALPACA_SECRET, ALPACA_ENDPOINT

# Optional: import Alpaca SDK if available
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    StockHistoricalDataClient = None

# Yahoo Finance fetch
def YF_fetch_data(ticker, start_date=None, end_date=None):
    try:
        if start_date and end_date:
            print(f"[YF_fetch_data] Fetching {ticker} from {start_date} to {end_date} at interval {INTERVAL}")
            data = yf.download(ticker, start=start_date, end=end_date, interval=INTERVAL)
        else:
            fallback_days = INTERVAL_LOOKBACK_LIMITS.get(INTERVAL, 30)
            fallback_period = f"{fallback_days}d"
            print(f"[YF_fetch_data] Fetching {ticker} using fallback period: {fallback_period} at interval {INTERVAL}")
            data = yf.download(ticker, period=fallback_period, interval=INTERVAL)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        data.reset_index(inplace=True)

        print(f"\n[YF] Data for {ticker} ({len(data)} rows):")
        print(data.head(5))
        print(data.tail(2))

        return data

    except Exception as e:
        logging.error(f"[YF_fetch_data] Error fetching data for {ticker}: {e}")
        return None

# Alpaca fetch
def Alpaca_fetch_data(ticker, start_date, end_date, alpaca_key="", alpaca_secret="", endpoint=ALPACA_ENDPOINT):


    if StockHistoricalDataClient is None:
        logging.error("Alpaca SDK not installed.")
        return None

    try:
        print(f"[Alpaca_fetch_data] Fetching {ticker} from {start_date} to {end_date} at interval {INTERVAL}")

        # TimeFrame map
        tf_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, "Minute"),
            "15m": TimeFrame(15, "Minute"),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day
        }

        timeframe = tf_map.get(INTERVAL, TimeFrame.Hour)

        client = StockHistoricalDataClient(
            api_key=alpaca_key,
            secret_key=alpaca_secret
        )

        request_params = StockBarsRequest(symbol_or_symbols=ticker, timeframe=timeframe, start=start_date, end=end_date)
        bars = client.get_stock_bars(request_params).df

        if bars.empty:
            logging.warning(f"[Alpaca] No data returned for {ticker}")
            return None

        df = bars.reset_index()
        df = df[df['symbol'] == ticker]  # Filter by symbol (multi-index support)
        df.drop(columns=['symbol'], inplace=True)

        df.rename(columns={
            'timestamp': 'Datetime',
            'open': f'Open_{ticker}',
            'high': f'High_{ticker}',
            'low': f'Low_{ticker}',
            'close': f'Close_{ticker}',
            'volume': f'Volume_{ticker}'
        }, inplace=True)

        print(f"\n[Alpaca] Data for {ticker} ({len(df)} rows):")
        print(df.head(5))
        print(df.tail(2))

        return df

    except Exception as e:
        logging.error(f"[Alpaca_fetch_data] Error fetching data for {ticker}: {e}")
        return None





def fetch_data(ticker, start_date=None, end_date=None, **kwargs):
    if DATA_SOURCE == "YF":
        return YF_fetch_data(ticker, start_date=start_date, end_date=end_date)
    elif DATA_SOURCE == "Alpaca":
        return Alpaca_fetch_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            alpaca_key=kwargs.get("alpaca_key", ALPACA_KEY),
            alpaca_secret=kwargs.get("alpaca_secret", ALPACA_SECRET),
            endpoint=kwargs.get("endpoint", ALPACA_ENDPOINT)
        )
    else:
        logging.error(f"Unsupported DATA_SOURCE: {DATA_SOURCE}")
        return None
