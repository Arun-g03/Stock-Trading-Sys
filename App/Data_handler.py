# data_handler.py

import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from App.Logger import System_Log
from Config.Config import Alpaca_API_KEY, Alpaca_SECRET_KEY

# Setup the logger
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
            raise

    @staticmethod
    def load_from_yfinance(ticker, start_date, end_date, interval='1d'):
        """
        Load historical data from Yahoo Finance with proper column handling.
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if data.empty:
                raise ValueError("No data received from Yahoo Finance. Check API requests.")
            
            # Flatten MultiIndex if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Ensure 'Close' column exists
            if "Close" not in data.columns:
                if "Adj Close" in data.columns:
                    data.rename(columns={"Adj Close": "Close"}, inplace=True)
                else:
                    raise KeyError("Neither 'Close' nor 'Adj Close' columns found in Yahoo Finance data.")
            
            data.reset_index(inplace=True)
            system_logger.info(f"Data loaded successfully from Yahoo Finance for {ticker} from {start_date} to {end_date}")
            system_logger.info(data)
            return data
        except Exception as e:
            system_logger.error(f"Error loading data from Yahoo Finance: {e}")
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
            raise
    
    @staticmethod
    def run(ticker, start_date, end_date, data_source='yfinance', api_key=None, api_secret=None, base_url=None):
        """
        Run the data handler to fetch data from the selected source.
        """
        try:
            if data_source == 'yfinance':
                return DataHandler.load_from_yfinance(ticker, start_date, end_date)
            elif data_source == 'alpaca':
                if not all([api_key, api_secret, base_url]):
                    raise ValueError("Missing Alpaca API credentials.")
                return DataHandler.load_from_alpaca(ticker, start_date, end_date, api_key, api_secret, base_url)
            else:
                raise ValueError("Invalid data source. Choose 'yfinance' or 'alpaca'.")
        except Exception as e:
            system_logger.error(f"Error running data handler: {e}")
            raise
