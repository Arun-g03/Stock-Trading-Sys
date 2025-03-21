# forecasts/arima_forecast.py

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ARIMAForecast:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.warmup_predictions = None
        self.warmup_actuals = None
        self.metrics = {'mse': None, 'mae': None}

    def train(self, data: pd.DataFrame):
        try:
            if len(data) > 0:
                self.model = ARIMA(data['Close'], order=self.order).fit()
            else:
                logging.warning("ARIMA: Not enough data to train model.")
                self.model = None
        except Exception as e:
            logging.error(f"Error training ARIMA model: {e}")
            self.model = None

    def predict(self, data: pd.DataFrame, steps=10):
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            return self.model.forecast(steps=steps)
        except Exception as e:
            logging.error(f"Error making ARIMA predictions: {e}")
            return np.zeros(steps)

    def perform_warmup(self, data: pd.DataFrame, warmup_period):
        try:
            warmup_data = data.iloc[-warmup_period:]
            self.warmup_actuals = warmup_data['Close'].values
            warmup_preds = []
            for i in range(warmup_period):
                train_data = data.iloc[:(-warmup_period + i)]
                self.train(train_data)
                if self.model is not None:
                    pred = self.predict(train_data, steps=1)[0]
                else:
                    pred = data.iloc[-warmup_period + i - 1]['Close'] if i > 0 else data.iloc[-warmup_period - 1]['Close']
                warmup_preds.append(pred)
            self.warmup_predictions = np.array(warmup_preds)
            self.metrics['mse'] = mean_squared_error(self.warmup_actuals, self.warmup_predictions)
            self.metrics['mae'] = mean_absolute_error(self.warmup_actuals, self.warmup_predictions)
            logging.info(f"ARIMA Warmup metrics - MSE: {self.metrics['mse']:.4f}, MAE: {self.metrics['mae']:.4f}")
        except Exception as e:
            logging.error(f"Error during ARIMA warmup: {e}")
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
            logging.error(f"Error in ARIMA forecast: {e}")
            return np.zeros(steps)
