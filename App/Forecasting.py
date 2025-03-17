import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from App.Logger import System_Log

# Setup the logger
system_logger = System_Log.setup_logger('forecasting_model')

# Base class to define the common interface
class BaseForecast:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def train(self, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict(self, data, steps=10):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def forecast(self, data, steps=10):
        # For many models, we train and then predict. Override if different logic is needed.
        self.train(data)
        return self.predict(data, steps=steps)

# LSTM forecasting class
class LSTMForecast(BaseForecast):
    def __init__(self, epochs=50, hidden_size=64, learning_rate=0.001):
        super().__init__()
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.model = None

    def train(self, data):
        try:
            if data.isnull().sum().any():
                raise ValueError("Input data contains missing values.")

            # Prepare data
            scaled_data = self.scaler.fit_transform(data[['Close']].values.reshape(-1, 1))
            train_data = torch.FloatTensor(scaled_data[:-1])
            target_data = torch.FloatTensor(scaled_data[1:])

            # Define LSTM model
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=self.hidden_size, output_size=1):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1])
            
            self.model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(train_data.unsqueeze(1))
                loss = criterion(output, target_data.unsqueeze(1))
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    system_logger.info(f"LSTM Epoch {epoch}/{self.epochs}, Loss: {loss.item()}")
        except Exception as e:
            system_logger.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, data, steps=10):
        try:
            # For one-step-ahead predictions on available data
            with torch.no_grad():
                input_data = torch.FloatTensor(self.scaler.transform(data[['Close']].values.reshape(-1, 1))).unsqueeze(1)
                predictions = self.model(input_data).numpy()
                return self.scaler.inverse_transform(predictions)
        except Exception as e:
            system_logger.error(f"Error making LSTM predictions: {e}")
            raise

# GRU forecasting class
class GRUForecast(BaseForecast):
    def __init__(self, epochs=50, hidden_size=64, learning_rate=0.001):
        super().__init__()
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.model = None

    def train(self, data):
        try:
            if data.isnull().sum().any():
                raise ValueError("Input data contains missing values.")

            # Prepare data
            scaled_data = self.scaler.fit_transform(data[['Close']].values.reshape(-1, 1))
            train_data = torch.FloatTensor(scaled_data[:-1])
            target_data = torch.FloatTensor(scaled_data[1:])

            # Define GRU model
            class GRUModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=self.hidden_size, output_size=1):
                    super(GRUModel, self).__init__()
                    self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.fc(out[:, -1])
            
            self.model = GRUModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(train_data.unsqueeze(1))
                loss = criterion(output, target_data.unsqueeze(1))
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    system_logger.info(f"GRU Epoch {epoch}/{self.epochs}, Loss: {loss.item()}")
        except Exception as e:
            system_logger.error(f"Error training GRU model: {e}")
            raise

    def predict(self, data, steps=10):
        try:
            with torch.no_grad():
                input_data = torch.FloatTensor(self.scaler.transform(data[['Close']].values.reshape(-1, 1))).unsqueeze(1)
                predictions = self.model(input_data).numpy()
                return self.scaler.inverse_transform(predictions)
        except Exception as e:
            system_logger.error(f"Error making GRU predictions: {e}")
            raise



class ARIMAForecast:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def train(self, data):
        try:
            # Ensure Date is a datetime index
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            
            # Ensure the index is a proper time series frequency
            data = data.asfreq('B')  # Set to business-day frequency

            self.model = ARIMA(data['Close'], order=self.order).fit()
            system_logger.info("ARIMA model trained successfully.")
        except Exception as e:
            system_logger.error(f"Error training ARIMA model: {e}")
            raise

    def predict(self, steps=10):
        try:
            forecast = self.model.forecast(steps=steps)
            return forecast
        except Exception as e:
            system_logger.error(f"Error making ARIMA predictions: {e}")
            raise

    def forecast(self, data, steps=10):
        self.train(data)
        return self.predict(steps=steps)


# Holt–Winters forecasting class
class HoltWintersForecast(BaseForecast):
    def __init__(self, seasonal_periods=12):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def train(self, data):
        try:
            self.model = ExponentialSmoothing(
                data['Close'],
                trend='add',
                seasonal='add',
                seasonal_periods=self.seasonal_periods
            ).fit()
            system_logger.info("Holt–Winters model trained successfully.")
        except Exception as e:
            system_logger.error(f"Error training Holt–Winters model: {e}")
            raise

    def predict(self, data, steps=10):
        try:
            forecast = self.model.forecast(steps=steps)
            return forecast
        except Exception as e:
            system_logger.error(f"Error making Holt–Winters predictions: {e}")
            raise

    def forecast(self, data, steps=10):
        self.train(data)
        return self.predict(data, steps=steps)

# Linear Regression forecasting class
class LinearRegressionForecast(BaseForecast):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.model = None

    def train(self, data):
        try:
            X, y = [], []
            series = data['Close'].values
            for i in range(len(series) - self.window_size):
                X.append(series[i:i + self.window_size])
                y.append(series[i + self.window_size])
            X = np.array(X)
            y = np.array(y)
            self.model = LinearRegression().fit(X, y)
            system_logger.info("Linear Regression model trained successfully.")
        except Exception as e:
            system_logger.error(f"Error training Linear Regression model: {e}")
            raise

    def predict(self, data, steps=10):
        try:
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
            system_logger.error(f"Error making Linear Regression predictions: {e}")
            raise

    def forecast(self, data, steps=10):
        self.train(data)
        return self.predict(data, steps=steps)

# Factory to instantiate the appropriate forecaster
class ForecastingFactory:
    @staticmethod
    def get_forecaster(method="LSTM", **kwargs):
        if method == "LSTM":
            return LSTMForecast(**kwargs)
        elif method == "GRU":
            return GRUForecast(**kwargs)
        elif method == "ARIMA":
            return ARIMAForecast(**kwargs)
        elif method == "HoltWinters":
            return HoltWintersForecast(**kwargs)
        elif method == "LinearRegression":
            return LinearRegressionForecast(**kwargs)
        else:
            raise ValueError("Invalid forecasting method selected. "
                             "Choose one of 'LSTM', 'GRU', 'ARIMA', 'HoltWinters', or 'LinearRegression'.")

