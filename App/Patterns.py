# patterns.py

import pandas as pd
import numpy as np
import ta
from Logger import System_Log
import traceback

# Setup the logger
system_logger = System_Log.setup_logger('patterns')

class Patterns:
    @staticmethod
    def higher_highs_lower_lows(data, window=20):
        """
        Identify Higher Highs and Lower Lows in the data.
        """
        try:
            data['high_max'] = data['High'].rolling(window=window).max()
            data['low_min'] = data['Low'].rolling(window=window).min()
            data['higher_highs'] = data['High'] > data['high_max'].shift(1)
            data['lower_lows'] = data['Low'] < data['low_min'].shift(1)
            system_logger.info("Higher Highs and Lower Lows identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Higher Highs and Lower Lows: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def double_top(data, window=20):
        """
        Identify Double Top pattern in the data.
        """
        try:
            data['high_max'] = data['High'].rolling(window=window).max()
            data['double_top'] = (data['High'] == data['high_max']) & (data['High'].shift(window) == data['high_max'])
            system_logger.info("Double Top pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Double Top pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def head_and_shoulders(data, window=20):
        """
        Identify Head and Shoulders pattern in the data.
        """
        try:
            data['head_and_shoulders'] = ((data['High'].shift(2*window) < data['High'].shift(window)) &
                                          (data['High'] > data['High'].shift(window)) &
                                          (data['High'] > data['High'].shift(2*window)) &
                                          (data['High'].shift(window) < data['High'].shift(2*window)) &
                                          (data['High'] < data['High'].shift(2*window)))
            system_logger.info("Head and Shoulders pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Head and Shoulders pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def triple_bottom(data, window=20):
        """
        Identify Triple Bottom pattern in the data.
        """
        try:
            data['low_min'] = data['Low'].rolling(window=window).min()
            data['triple_bottom'] = ((data['Low'] == data['low_min']) &
                                     (data['Low'].shift(window) == data['low_min']) &
                                     (data['Low'].shift(2*window) == data['low_min']))
            system_logger.info("Triple Bottom pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Triple Bottom pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def cup_and_handle(data, window=20):
        """
        Identify Cup and Handle pattern in the data.
        """
        try:
            data['cup'] = (data['High'].rolling(window).max() - data['Low'].rolling(window).min())
            data['handle'] = (data['High'].shift(window//2).rolling(window).max() - data['Low'].shift(window//2).rolling(window).min())
            data['cup_and_handle'] = data['handle'] < data['cup']
            system_logger.info("Cup and Handle pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Cup and Handle pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def bullish_engulfing(data):
        """
        Identify Bullish Engulfing pattern in the data.
        """
        try:
            data['bullish_engulfing'] = ((data['Open'].shift(1) > data['Close'].shift(1)) &
                                         (data['Open'] < data['Close']) &
                                         (data['Open'] < data['Close'].shift(1)) &
                                         (data['Close'] > data['Open'].shift(1)))
            system_logger.info("Bullish Engulfing pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Bullish Engulfing pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def bearish_engulfing(data):
        """
        Identify Bearish Engulfing pattern in the data.
        """
        try:
            data['bearish_engulfing'] = ((data['Open'].shift(1) < data['Close'].shift(1)) &
                                         (data['Open'] > data['Close']) &
                                         (data['Open'] > data['Close'].shift(1)) &
                                         (data['Close'] < data['Open'].shift(1)))
            system_logger.info("Bearish Engulfing pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Bearish Engulfing pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def morning_star(data):
        """
        Identify Morning Star pattern in the data.
        """
        try:
            data['morning_star'] = ((data['Close'].shift(2) < data['Open'].shift(2)) &
                                    (data['Close'].shift(1) < data['Open'].shift(1)) &
                                    (data['Close'] > data['Open']))
            system_logger.info("Morning Star pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Morning Star pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def evening_star(data):
        """
        Identify Evening Star pattern in the data.
        """
        try:
            data['evening_star'] = ((data['Close'].shift(2) > data['Open'].shift(2)) &
                                    (data['Close'].shift(1) > data['Open'].shift(1)) &
                                    (data['Close'] < data['Open']))
            system_logger.info("Evening Star pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Evening Star pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def hammer(data):
        """
        Identify Hammer pattern in the data.
        """
        try:
            body_size = abs(data['Close'] - data['Open'])
            lower_shadow = data['Open'] - data['Low']
            data['hammer'] = (lower_shadow > body_size * 2) & (data['Close'] > data['Open'])
            system_logger.info("Hammer pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Hammer pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def shooting_star(data):
        """
        Identify Shooting Star pattern in the data.
        """
        try:
            body_size = abs(data['Close'] - data['Open'])
            upper_shadow = data['High'] - data['Close']
            data['shooting_star'] = (upper_shadow > body_size * 2) & (data['Open'] > data['Close'])
            system_logger.info("Shooting Star pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Shooting Star pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def rsi_divergence(data, window=14):
        """
        Identify RSI Divergence in the data.
        """
        try:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
            data['price_diff'] = data['Close'].diff(window)
            data['rsi_diff'] = data['RSI'].diff(window)
            data['rsi_divergence'] = (data['price_diff'] * data['rsi_diff'] < 0)
            system_logger.info("RSI Divergence identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying RSI Divergence: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def bollinger_band_squeeze(data, window=20, std_dev=2):
        """
        Identify Bollinger Band Squeeze in the data.
        """
        try:
            bb = ta.volatility.BollingerBands(data['Close'], window=window, window_dev=std_dev)
            data['bb_squeeze'] = (bb.bollinger_hband() - bb.bollinger_lband()) / data['Close']
            data['bollinger_band_squeeze'] = data['bb_squeeze'] < (std_dev / 2)
            system_logger.info("Bollinger Band Squeeze identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Bollinger Band Squeeze: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def moving_average_crossover(data, short_window=50, long_window=200):
        """
        Identify Moving Average Crossover in the data.
        """
        try:
            # Calculate short and long moving averages
            data['short_ma'] = ta.trend.SMAIndicator(data['Close'], window=short_window).sma_indicator()
            data['long_ma'] = ta.trend.SMAIndicator(data['Close'], window=long_window).sma_indicator()

            # Identify crossover (boolean column)
            data['ma_crossover'] = data['short_ma'] > data['long_ma']

            # Ensure correct boolean type before applying bitwise NOT (~)
            data['ma_crossover_signal'] = data['ma_crossover'].astype(bool) & ~data['ma_crossover'].shift(1).infer_objects(copy=False).astype(bool).fillna(False)



            system_logger.info("Moving Average Crossover identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Moving Average Crossover: {e}")
            traceback.print_exc()
            raise


    @staticmethod
    
    def adx_trend_strength(data, window=14):
        """
        Identify ADX Trend Strength in the data.
        """
        try:
            if len(data) < window:
                data['adx'] = np.nan
                data['strong_trend'] = False
                return data

            adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window)
            data['adx'] = adx.adx()
            data['strong_trend'] = data['adx'] > 25
            system_logger.info("ADX Trend Strength identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying ADX Trend Strength: {e}")
            traceback.print_exc()
            
            raise


    @staticmethod
    def stochastic_oscillator(data, window=14):
        """
        Identify Stochastic Oscillator in the data.
        """
        try:
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=window)
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            data['stoch_overbought'] = data['stoch_k'] > 80
            data['stoch_oversold'] = data['stoch_k'] < 20
            system_logger.info("Stochastic Oscillator identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Stochastic Oscillator: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def pennant(data, window=20):
        """
        Identify Pennant pattern in the data.
        """
        try:
            data['pennant'] = ((data['Close'] > data['Close'].shift(window)) & 
                               (data['Close'] < data['Close'].shift(window * 2)) &
                               (data['High'] < data['High'].shift(window)) &
                               (data['Low'] > data['Low'].shift(window)))
            system_logger.info("Pennant pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Pennant pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def flag(data, window=20):
        """
        Identify Flag pattern in the data.
        """
        try:
            data['flag'] = ((data['Close'] > data['Close'].shift(window)) &
                            (data['Close'] < data['Close'].shift(window * 2)) &
                            (data['High'] < data['High'].shift(window)) &
                            (data['Low'] > data['Low'].shift(window)) &
                            (data['Volume'] > data['Volume'].shift(window)))
            system_logger.info("Flag pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Flag pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def wedge(data, window=20):
        """
        Identify Wedge pattern in the data.
        """
        try:
            data['wedge'] = ((data['High'].rolling(window).max() - data['Low'].rolling(window).min()) /
                             (data['High'].rolling(window * 2).max() - data['Low'].rolling(window * 2).min()) < 0.5)
            system_logger.info("Wedge pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Wedge pattern: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def triangle(data, window=20):
        """
        Identify Triangle pattern in the data.
        """
        try:
            data['high_trend'] = data['High'].rolling(window).max()
            data['low_trend'] = data['Low'].rolling(window).min()
            data['triangle'] = (data['High'] < data['high_trend']) & (data['Low'] > data['low_trend'])
            system_logger.info("Triangle pattern identified successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error identifying Triangle pattern: {e}")
            traceback.print_exc()
            raise

# Example usage:
# data = pd.read_csv('path_to_your_csv')
# data = Patterns.higher_highs_lower_lows(data)
# data = Patterns.double_top(data)
# data = Patterns.head_and_shoulders(data)
# data = Patterns.triple_bottom(data)
# data = Patterns.cup_and_handle(data)
# data = Patterns.bullish_engulfing(data)
# data = Patterns.bearish_engulfing(data)
# data = Patterns.morning_star(data)
# data = Patterns.evening_star(data)
# data = Patterns.hammer(data)
# data = Patterns.shooting_star(data)
# data = Patterns.rsi_divergence(data)
# data = Patterns.bollinger_band_squeeze(data)
# data = Patterns.moving_average_crossover(data)
# data = Patterns.adx_trend_strength(data)
# data = Patterns.stochastic_oscillator(data)
# data = Patterns.pennant(data)
# data = Patterns.flag(data)
# data = Patterns.wedge(data)
# data = Patterns.triangle(data)
