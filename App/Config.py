# config/config.py

# Set log path
LOG_PATH = 'logs/system.log'


# Time Intervals Configuration
INTERVALS = {
    'minute': ['1m', '5m', '15m', '30m'],
    'hourly': ['1h', '4h'],
    'daily': ['1d'],
    'weekly': ['1wk'],
    'monthly': ['1mo']
}


TRADE_DATA_SOURCE = 'yfinance' #yfinance or alpaca


# WindowConfig.py

class WindowConfig:
    """
    Utility class to configure window sizes based on data interval.
    Automatically adjusts technical indicator parameters for different timeframes.
    """

    # Default multipliers for different intervals relative to daily data
    INTERVAL_MULTIPLIERS = {
        '1m': 1/390,     # 1 minute (390 minutes in a trading day)
        '5m': 5/390,     # 5 minutes
        '15m': 15/390,   # 15 minutes
        '30m': 30/390,   # 30 minutes
        '1h': 1/6.5,     # 1 hour (6.5 trading hours in a day)
        '4h': 4/6.5,     # 4 hours
        '1d': 1.0,       # 1 day (baseline)
        '1wk': 5.0,      # 1 week (5 trading days)
        '1mo': 21.0      # 1 month (approx 21 trading days)
    }

    # Default window sizes for daily data
    DEFAULT_WINDOWS = {
        'rsi': 14,
        'macd_slow': 26,
        'macd_fast': 12,
        'macd_signal': 9,
        'atr': 14,
        'bollinger': 20,
        'ma': 20,
        'ema': 20,
        'adx': 14,
        'stochastic': 14,
        'volatility': 20,
        'support_resistance': 20,
        'volume_ma': 20,
        'pattern_short': 20,
        'pattern_medium': 50,
        'pattern_long': 100,
        'ma_crossover_short': 50,
        'ma_crossover_long': 200
    }

    @classmethod
    def get_window(cls, window_type, interval='1d', min_window=2):
        """
        Get the appropriate window size for a given indicator type and data interval.

        Args:
            window_type (str): Type of window/indicator (e.g., 'rsi', 'macd_slow')
            interval (str): Data interval (e.g., '1m', '5m', '1d')
            min_window (int): Minimum window size to return

        Returns:
            int: Adjusted window size for the given interval
        """
        # Get the default window size for this indicator
        default_window = cls.DEFAULT_WINDOWS.get(window_type, 14)

        # Get the multiplier for this interval
        multiplier = cls.INTERVAL_MULTIPLIERS.get(interval, 1.0)

        # Calculate adjusted window size
        adjusted_window = max(min_window, int(default_window * multiplier))

        # Ensure a minimum usable window for very small intervals
        if interval in ['1m', '5m'] and adjusted_window < min_window:
            adjusted_window = min_window

        return adjusted_window

    @classmethod
    def get_all_windows(cls, interval='1d'):
        """
        Get all window sizes adjusted for a specific interval.

        Args:
            interval (str): Data interval (e.g., '1m', '5m', '1d')

        Returns:
            dict: Dictionary of all adjusted window sizes
        """
        return {
            window_type: cls.get_window(window_type, interval)
            for window_type in cls.DEFAULT_WINDOWS
        }
