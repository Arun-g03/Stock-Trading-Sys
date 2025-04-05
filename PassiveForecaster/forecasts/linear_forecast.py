# forecasts/linear_regression_forecast.py

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import traceback

class LinearRegressionForecast:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.model = None
        self.warmup_predictions = None
        self.warmup_actuals = None
        self.metrics = {'mse': None, 'mae': None}

    def train(self, data: pd.DataFrame):
        try:
            X, y = [], []
            series = data['Close'].values
            if series.ndim > 1:
                series = series.squeeze()

            if len(series) <= self.window_size:
                logging.warning(f"Not enough data for window size {self.window_size}. Adjusting window size.")
                self.window_size = max(1, len(series) - 1)

            for i in range(len(series) - self.window_size):
                X.append(series[i:i + self.window_size])
                y.append(series[i + self.window_size])

            if len(X) > 0:
                X_array = np.array(X)
                if X_array.ndim == 3:
                    X_array = X_array.reshape(X_array.shape[0], X_array.shape[1])
                X_array = X_array.reshape(len(X), self.window_size)
                y_array = np.array(y)
                self.model = LinearRegression().fit(X_array, y_array)
            else:
                logging.warning("Not enough data to train Linear Regression model")
                self.model = None
        except Exception as e:
            logging.error(f"Error training Linear Regression model: {e}")
            self.model = None

    def predict(self, data: pd.DataFrame, steps=10):
        try:
            if self.model is None:
            traceback.print_exc()
            raise ValueError("Model not trained")
            series = data['Close'].values
            predictions = []
            input_seq = list(series[-self.window_size:])
            for _ in range(steps):
                input_arr = np.array(input_seq[-self.window_size:]).reshape(1, -1)
                pred = self.model.predict(input_arr)[0]
                predictions.append(pred)
                input_seq.append(pred)
            return np.array(predictions)
        except Exception as e:
            logging.error(f"Error making Linear Regression predictions: {e}")
            return np.zeros(steps)

    def perform_warmup(self, data: pd.DataFrame, warmup_period):
        try:
            warmup_data = data.iloc[-warmup_period:]
            self.warmup_actuals = warmup_data['Close'].values
            warmup_preds = []
            for i in range(warmup_period):
                train_data = data.iloc[:-warmup_period + i]
                self.train(train_data)
                if self.model is not None and len(train_data) > self.window_size:
                    window_data = train_data['Close'].values[-self.window_size:]
                    input_arr = window_data.reshape(1, -1)
                    pred = self.model.predict(input_arr)[0]
                    warmup_preds.append(pred)
                else:
                    prev_value = data.iloc[-warmup_period + i - 1]['Close'] if i > 0 else data.iloc[-warmup_period - 1]['Close']
                    warmup_preds.append(prev_value)
            self.warmup_predictions = np.array(warmup_preds)
            self.metrics['mse'] = np.nan if len(self.warmup_actuals)==0 else \
                np.mean((self.warmup_actuals - self.warmup_predictions)**2)
            self.metrics['mae'] = np.nan if len(self.warmup_actuals)==0 else \
                np.mean(np.abs(self.warmup_actuals - self.warmup_predictions))
            logging.info(f"LinearRegression Warmup metrics - MSE: {self.metrics['mse']:.4f}, MAE: {self.metrics['mae']:.4f}")
        except Exception as e:
            logging.error(f"Error during LinearRegression warmup: {str(e)}")
            self.warmup_predictions = np.zeros(warmup_period)
            self.metrics['mse'] = float('nan')
            self.metrics['mae'] = float('nan')

    def forecast(self, data: pd.DataFrame, steps=10, warmup=0):
        try:
            if warmup > 0:
                self.train(data)  # Train on full data, including warmup
                self.perform_warmup(data, warmup)
                return self.predict(data, steps)  # Predict from real last data point
            else:
                self.train(data)
                return self.predict(data, steps)
        except Exception as e:
            logging.error(f"Error in Linear Regression forecast: {e}")
            return np.zeros(steps)
