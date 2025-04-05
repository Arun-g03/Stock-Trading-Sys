import pandas as pd
from Logger import System_Log
import traceback

system_logger = System_Log.setup_logger('Screener')

class StockScreener:
    def __init__(self, min_market_cap: float, min_volume: int, min_volatility: float):
        """Initialise screener with fundamental and technical thresholds."""
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.min_volatility = min_volatility
        self.data = None
    
    # 🔹 Fundamental Screening
    def filter_by_market_cap(self, data, min_cap):
        """
        Filter stocks based on market capitalization, but handle missing column.
        """
        if 'market_cap' not in data.columns:
            system_logger.warning("market_cap column is missing. Skipping market cap filter.")
            return data  # Return unfiltered data if column is missing

        return data[data['market_cap'] >= min_cap]

    
    def filter_by_volume(self, data: pd.DataFrame, min_volume: int) -> pd.DataFrame:
        """Filter out stocks with low trading volume."""
        return data[data['Volume'] >= min_volume]
    
    def filter_by_volatility(self, data: pd.DataFrame, min_volatility: float) -> pd.DataFrame:
        """Remove stocks with very low price movement."""
        return data[data['volatility'] >= min_volatility]
    
    # 🔹 Technical Screening (Short-Term)
    def apply_momentum_filters(self, data):
        """
        Filter short-term trading candidates based on momentum indicators.
        """
        # Ensure correct column name
        rsi_column = 'RSI' if 'RSI' in data.columns else ('rsi_14' if 'rsi_14' in data.columns else None)

        if rsi_column is None:
            system_logger.warning("RSI column is missing. Skipping RSI filter.")
            return data  # Return unfiltered data if RSI is missing

        # Apply RSI filter (oversold/overbought)
        data = data[data[rsi_column] <= 70]

        # Apply MACD filter (positive momentum)
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            data = data[data['MACD'] > data['MACD_Signal']]
        else:
            system_logger.warning("MACD columns missing. Skipping MACD filter.")

        return data

    
    def apply_breakout_filters(self, data):
        """
        Identify stocks breaking out of key levels for short-term trading.
        """
        if 'resistance' not in data.columns:
            system_logger.warning("Resistance column is missing. Skipping breakout filter.")
            return data  # Return unfiltered data if column is missing

        # Filter for stocks breaking above resistance
        data = data[data['Close'] > data['resistance']]
        
        # Confirm breakout with volume surge
        if 'Volume' in data.columns and 'Volume_ma_20' in data.columns:
            data = data[data['Volume'] > data['Volume_ma_20'] * 1.5]
        else:
            system_logger.warning("Volume moving average column is missing. Skipping volume confirmation filter.")

        return data

    
    # 🔹 Technical Screening (Long-Term)
    def apply_trend_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter long-term investment candidates based on moving averages and money flow."""
        # Price above major moving averages
        data = data[
            (data['Close'] > data['ma_50']) & 
            (data['Close'] > data['ma_200'])
        ]
        # Positive money flow
        data = data[data['mfi'] > 50]
        return data
    
    def apply_fundamental_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure strong financials for long-term candidates."""
        # Filter for positive earnings growth
        data = data[data['earnings_growth'] > 0]
        # Filter for healthy debt ratios
        data = data[data['debt_to_equity'] < 2.0]
        return data
    
    # 🔹 Final Screening
    def screen_short_term_candidates(self, data, tickers):
        """
        Run the full short-term screening process and return a dictionary of screened tickers.
        """
        screened_tickers = {}

        for ticker in tickers:
            if ticker not in data:
                system_logger.warning(f"Data for {ticker} is missing. Skipping.")
                continue  # Skip if ticker data is missing

            stock_data = data[ticker].copy()

            # Apply fundamental filters
            stock_data = self.filter_by_market_cap(stock_data, self.min_market_cap)
            stock_data = self.filter_by_volume(stock_data, self.min_volume)
            stock_data = self.filter_by_volatility(stock_data, self.min_volatility)

            # Apply technical filters for short-term
            stock_data = self.apply_momentum_filters(stock_data)
            stock_data = self.apply_breakout_filters(stock_data)

            # Store the result (True if stock passes filters, False otherwise)
            screened_tickers[ticker] = not stock_data.empty

        return screened_tickers


    
    def screen_long_term_candidates(self, data, tickers):
        """
        Run the full long-term screening process and return a dictionary of screened tickers.
        """
        screened_tickers = {}

        for ticker in tickers:
            if ticker not in data:
                system_logger.warning(f"Data for {ticker} is missing. Skipping.")
                continue  # Skip if ticker data is missing

            stock_data = data[ticker].copy()

            # Apply fundamental filters
            stock_data = self.filter_by_market_cap(stock_data, self.min_market_cap)
            stock_data = self.filter_by_volume(stock_data, self.min_volume)
            stock_data = self.apply_fundamental_filters(stock_data)

            # Apply technical filters for long-term investment
            stock_data = self.apply_trend_filters(stock_data)

            # Store the result (True if stock passes filters, False otherwise)
            screened_tickers[ticker] = not stock_data.empty

        return screened_tickers

    
    def run(self, data, tickers):
        """
        Run both short-term and long-term screening and return results per ticker.
        """
        short_term_candidates = self.screen_short_term_candidates(data, tickers)
        long_term_candidates = self.screen_long_term_candidates(data, tickers)

        return {
            "short_term": short_term_candidates,
            "long_term": long_term_candidates
        }
