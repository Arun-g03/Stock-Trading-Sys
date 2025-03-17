# signal_generator.py

import pandas as pd
import numpy as np
from App.Feature_engineering import FeatureEngineering
from App.model import Model
from sklearn.metrics import accuracy_score
from App.Logger import System_Log

# Setup the logger
system_logger = System_Log.setup_logger('signal_generator')

class SignalGenerator:
    @staticmethod
    def generate_rule_based_signals(data):
        """
        Generate trading signals based on patterns and indicators.
        """
        try:
            data['Rule_Signal'] = 0  # Default to no signal

            # Example: Buy when RSI < 30 and MACD > MACD Signal
            data.loc[(data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']), 'Rule_Signal'] = 1
            # Example: Sell when RSI > 70 and MACD < MACD Signal
            data.loc[(data['RSI'] > 70) & (data['MACD'] < data['MACD_Signal']), 'Rule_Signal'] = -1

            # Ensure the `Signal` column is created
            if 'Signal' not in data.columns:
                data['Signal'] = data['Rule_Signal']  # Assign rule-based signals as default

            system_logger.info("Rule-based signals generated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error generating rule-based signals: {e}")
            raise




    @staticmethod
    def generate_consensus_signal(data):
        """
        Generate consensus trading signal based on rule-based and model-based signals.
        """
        try:
            data['Consensus_Signal'] = data[['Rule_Signal', 'Model_Signal']].mean(axis=1)
            data['Consensus_Signal'] = data['Consensus_Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            system_logger.info("Consensus signal generated successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error generating consensus signal: {e}")
            raise

    @staticmethod
    def backtest(data, initial_balance=10000):
        """
        Backtest trading strategy based on generated signals.
        """
        try:
            # Ensure data is not empty
            if data.empty:
                system_logger.error("Backtesting failed: Data is empty.")
                raise ValueError("Backtesting requires a non-empty dataset.")

            # Ensure we have at least two rows
            if len(data) < 2:
                system_logger.error("Not enough data for backtesting.")
                raise ValueError("Backtesting requires at least two rows of data.")

            # Reset index to ensure sequential order
            data = data.reset_index(drop=True)

            balance = initial_balance
            position = 0  # 1 for long, -1 for short
            balance_history = []

            for index, row in data.iterrows():
                if index == 0:  # First row, no previous price available
                    balance_history.append(balance)
                    continue

                # Ensure previous_close is valid
                if index > 0 and index < len(data):  
                    previous_close = data['Close'].iloc[index - 1]
                else:
                    previous_close = row['Close']  # Default to current price if out of bounds

                # Execute buy/sell based on Consensus Signal
                if row['Consensus_Signal'] == 1:  # Buy signal
                    if position <= 0:
                        position = 1
                        balance -= row['Close']  # Buy at close price
                elif row['Consensus_Signal'] == -1:  # Sell signal
                    if position >= 0:
                        position = -1
                        balance += row['Close']  # Sell at close price

                balance += position * (row['Close'] - previous_close)
                balance_history.append(balance)

            data['Balance'] = balance_history
            system_logger.info("Backtesting completed successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error in backtesting: {e}")
            raise




    @staticmethod
    def evaluate_signals(data):
        """
        Evaluate the accuracy of signals.
        """
        try:
            rule_accuracy = accuracy_score(data['Rule_Signal'], data['Consensus_Signal'])
            model_accuracy = accuracy_score(data['Model_Signal'], data['Consensus_Signal'])

            system_logger.info(f"Rule-Based Signal Accuracy: {rule_accuracy:.2f}")
            system_logger.info(f"Model-Based Signal Accuracy: {model_accuracy:.2f}")

            return rule_accuracy, model_accuracy
        except Exception as e:
            system_logger.error(f"Error evaluating signals: {e}")
            raise

# Example usage:
# data = pd.read_csv('path_to_your_csv')
# data = FeatureEngineering.engineer_features(data)
# data = SignalGenerator.generate_rule_based_signals(data)
# model, model_accuracy = Model.train_model(data)
# Model.save_model(model, 'path_to_save_model')
# model = Model.load_model('path_to_save_model')
# data = Model.apply_model(data, model)
# data = SignalGenerator.generate_consensus_signal(data)
# data = SignalGenerator.backtest(data)
# SignalGenerator.evaluate_signals(data)
# print(data.head())
