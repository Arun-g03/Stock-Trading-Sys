# data_handler.py

import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from Logger import System_Log
from ALPACA_KEYS import Alpaca_API_KEY, Alpaca_SECRET_KEY
import traceback
from Config import TRADE_DATA_SOURCE

system_logger = System_Log.setup_logger('data_handler')
class DataHandler:
    @staticmethod
    def load_from_csv(file_path):
        """
        Load historical data from a CSV file.
        """
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'])
            system_logger.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            system_logger.error(f"Error loading data from CSV: {e}")
            traceback.print_exc()
            raise


class DataHandler:
    @staticmethod
    def load_from_yfinance(ticker, start_date, end_date, interval='1d'):
        """
        Load historical data from Yahoo Finance with proper column handling.
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)

            # Print raw data if something looks wrong
            if data.empty:
                print(f"[DEBUG] Empty data returned:\n{data}")
                raise ValueError(f"No data received from Yahoo Finance for {ticker}. Check symbol and time range.")

            # Flatten MultiIndex if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            # Ensure 'Close' column exists
            if "Close" not in data.columns:
                if "Adj Close" in data.columns:
                    data.rename(columns={"Adj Close": "Close"}, inplace=True)
                else:
                    print(f"[DEBUG] Column names returned:\n{data.columns}")
                    raise KeyError(f"Neither 'Close' nor 'Adj Close' columns found for {ticker}.")

            data.reset_index(inplace=True)
            system_logger.info(f"Data loaded from Yahoo Finance for {ticker}: {start_date} → {end_date}")
            system_logger.info(data.tail())

            return data

        except Exception as e:
            system_logger.error(f"Error loading data for {ticker}: {e}")
            print(f"[EXCEPTION] Failed to load data for {ticker} from Yahoo:")
            traceback.print_exc()
            print("[DEBUG] Data content on failure:\n", data if 'data' in locals() else "No data object created.")
            raise


    @staticmethod
    def load_from_alpaca(ticker, start_date, end_date, api_key, api_secret, base_url):
        """
        Load historical data from Alpaca API.
        """
        try:
            api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            barset = api.get_barset(ticker, 'day', start=start_date, end=end_date)
            bars = barset[ticker]
            
            data = pd.DataFrame({
                'Date': [bar.t for bar in bars],
                'Open': [bar.o for bar in bars],
                'High': [bar.h for bar in bars],
                'Low': [bar.l for bar in bars],
                'Close': [bar.c for bar in bars],
                'Volume': [bar.v for bar in bars]
            })
            data['Date'] = pd.to_datetime(data['Date'])  # Ensure correct datetime format
            
            system_logger.info(f"Data loaded successfully from Alpaca API for {ticker} from {start_date} to {end_date}")
            return data
        except Exception as e:
            system_logger.error(f"Error loading data from Alpaca API: {e}")
            traceback.print_exc()
            raise
    
    @staticmethod
    def run(ticker, start_date, end_date, data_source=TRADE_DATA_SOURCE, api_key=None, api_secret=None, base_url=None):
        """
        Run the data handler to fetch data from the selected source.
        """
        try:
            if data_source == 'yfinance':
                return DataHandler.load_from_yfinance(ticker, start_date, end_date)
            elif data_source == 'alpaca':
                if not all([api_key, api_secret, base_url]):
                    traceback.print_exc()
                    raise ValueError("Missing Alpaca API credentials.")
                return DataHandler.load_from_alpaca(ticker, start_date, end_date, api_key, api_secret, base_url)
            else:
                traceback.print_exc()
            raise ValueError("Invalid data source. Choose 'yfinance' or 'alpaca'.")
        except Exception as e:
            system_logger.error(f"Error running data handler: {e}")
            traceback.print_exc()
            raise
