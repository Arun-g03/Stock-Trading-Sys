# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Patterns import Patterns
from Indicators import Indicators
from Logger import System_Log
import traceback

# Setup the logger
system_logger = System_Log.setup_logger('feature_engineering')

class FeatureEngineering:


    @staticmethod
    def calculate_volatility(data, window=20):
        """
        Calculate rolling volatility based on standard deviation of returns.
        """
        try:
            data = data.copy()  # Create a copy to avoid fragmentation
            returns = data['Close'].pct_change()  # Compute daily returns
            volatility = returns.rolling(window=window).std()  # Compute rolling volatility

            # Concatenate new columns to avoid fragmentation
            data = pd.concat([data, pd.DataFrame({'returns': returns, 'volatility': volatility})], axis=1)
            data.drop(columns=['returns'], inplace=True)  # Remove intermediate column

            system_logger.info(f"Volatility (window={window}) calculated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error calculating volatility: {e}")
            traceback.print_exc()
            raise


    @staticmethod
    def calculate_resistance(data, window=20):
        """
        Calculate resistance level based on the highest high over a rolling window.
        """
        try:
            data = data.copy()  # Prevent DataFrame fragmentation
            resistance = data['High'].rolling(window=window).max()  # Compute resistance levels

            # Concatenate to avoid fragmentation
            data = pd.concat([data, pd.DataFrame({'resistance': resistance})], axis=1)

            system_logger.info(f"Resistance levels (window={window}) calculated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error calculating resistance levels: {e}")
            traceback.print_exc()
            raise


    @staticmethod
    def calculate_support(data, window=20):
        """
        Calculate support level based on the lowest low over a rolling window.
        """
        try:
            data = data.copy()  # Prevent DataFrame fragmentation
            support = data['Low'].rolling(window=window).min()  # Compute support levels

            # Concatenate to avoid fragmentation
            data = pd.concat([data, pd.DataFrame({'support': support})], axis=1)

            system_logger.info(f"Support levels (window={window}) calculated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error calculating support levels: {e}")
            traceback.print_exc()
            raise


    @staticmethod
    def calculate_volume_moving_average(data, window=20):
        """
        Calculate the 20-day moving average of trading volume.
        """
        try:
            data = data.copy()  # Prevent DataFrame fragmentation
            volume_ma_20 = data['Volume'].rolling(window=window).mean()  # Compute volume moving average

            # Concatenate to avoid fragmentation
            data = pd.concat([data, pd.DataFrame({'Volume_ma_20': volume_ma_20})], axis=1)

            system_logger.info(f"Volume moving average (window={window}) calculated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error calculating volume moving average: {e}")
            traceback.print_exc()
            raise




    @staticmethod
    def add_patterns(data):
        """
        Add patterns as features to the data.
        """
        try:
            data = Patterns.higher_highs_lower_lows(data)
            data = Patterns.double_top(data)
            data = Patterns.head_and_shoulders(data)
            data = Patterns.triple_bottom(data)
            data = Patterns.cup_and_handle(data)
            data = Patterns.bullish_engulfing(data)
            data = Patterns.bearish_engulfing(data)
            data = Patterns.morning_star(data)
            data = Patterns.evening_star(data)
            data = Patterns.hammer(data)
            data = Patterns.shooting_star(data)
            data = Patterns.rsi_divergence(data)
            data = Patterns.bollinger_band_squeeze(data)
            data = Patterns.moving_average_crossover(data)
            data = Patterns.adx_trend_strength(data)
            data = Patterns.stochastic_oscillator(data)
            data = Patterns.pennant(data)
            data = Patterns.flag(data)
            data = Patterns.wedge(data)
            data = Patterns.triangle(data)
            system_logger.info("Patterns added successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error adding patterns: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def add_indicators(data):
        """
        Add indicators as features to the data.
        """
        try:
            data = Indicators.moving_average(data)
            data = Indicators.exponential_moving_average(data)
            data = Indicators.relative_strength_index(data)
            data = Indicators.bollinger_bands(data)
            data = Indicators.macd(data)
            data = Indicators.average_true_range(data)
            data = Indicators.stochastic_oscillator(data)
            data = Indicators.commodity_channel_index(data)
            data = Indicators.ichimoku_cloud(data)
            data = Indicators.aroon(data)
            data = Indicators.parabolic_sar(data)
            data = Indicators.volume_weighted_average_price(data)
            data = Indicators.on_balance_volume(data)
            data = Indicators.money_flow_index(data)
            data = Indicators.chaikin_money_flow(data)
            data = Indicators.ease_of_movement(data)
            data = Indicators.accumulation_distribution(data)
            data = Indicators.ultimate_oscillator(data)
            system_logger.info("Indicators added successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error adding indicators: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def handle_missing_values(data):
        """
        Handle missing values in the data.
        """
        try:
            data.ffill(inplace=True)
            data.bfill(inplace=True)

            system_logger.info("Missing values handled successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error handling missing values: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def normalise_data(data, columns):
        """
        Normalise specified columns in the data.
        """
        try:
            scaler = MinMaxScaler()
            data[columns] = scaler.fit_transform(data[columns])
            system_logger.info("Data normalised successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error normalising data: {e}")
            traceback.print_exc()
            raise

    
    @staticmethod
    def create_lagged_features(data, columns, lags=1):
        """
        Create lagged features for the specified columns.
        """
        try:
            lagged_dfs = []
            for lag in range(1, lags + 1):
                lagged_df = data[columns].shift(lag).add_suffix(f'_lag{lag}')
                lagged_dfs.append(lagged_df)

            # Use pd.concat to efficiently merge lagged features at once
            data = pd.concat([data] + lagged_dfs, axis=1)

            # De-fragment the DataFrame
            data = data.copy()

            system_logger.info("Lagged features created successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error creating lagged features: {e}")
            traceback.print_exc()
            raise


    @staticmethod
    def engineer_features(data):
        """
        Perform complete feature engineering on the data.
        """
        try:
            data = FeatureEngineering.add_patterns(data)
            data = FeatureEngineering.add_indicators(data)
            data = FeatureEngineering.handle_missing_values(data)

            feature_columns = [col for col in data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data = FeatureEngineering.create_lagged_features(data, feature_columns, lags=3)
            data = FeatureEngineering.normalise_data(data, feature_columns)

            # Drop rows with NaN values created by lagging
            data.dropna(inplace=True)

            system_logger.info("Feature engineering completed successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error in feature engineering: {e}")
            traceback.print_exc()
            raise

# Example usage:
# data = pd.read_csv('path_to_your_csv')
# data = FeatureEngineering.engineer_features(data)
# print(data.head())
