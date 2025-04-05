# forecasts/gru_forecast.py

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

class GRUForecast:
    def __init__(self, epochs=50, hidden_size=64, learning_rate=0.001):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.warmup_predictions = None
        self.warmup_actuals = None
        self.metrics = {'mse': None, 'mae': None}
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.model = None

    def train(self, data: pd.DataFrame):
        try:
            if data.isnull().sum().any():
            traceback.print_exc()
            raise ValueError("Input data contains missing values.")
            scaled_data = self.scaler.fit_transform(data[['Close']].values.reshape(-1, 1))
            train_data = torch.FloatTensor(scaled_data[:-1])
            target_data = torch.FloatTensor(scaled_data[1:])
            hidden_size = self.hidden_size

            class GRUModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=hidden_size, output_size=1):
                    super().__init__()
                    self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.fc(out[:, -1])

            self.model = GRUModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(train_data.unsqueeze(1))
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    logging.info(f"GRU Epoch {epoch}/{self.epochs}, Loss: {loss.item()}")
        except Exception as e:
            logging.error(f"Error training GRU model: {e}")

    def predict(self, data: pd.DataFrame, steps=10):
        try:
            with torch.no_grad():
                scaled_data = self.scaler.transform(data[['Close']].values.reshape(-1, 1))
                input_seq = torch.FloatTensor(scaled_data[-1:])
                predictions = []
                model_input = input_seq.unsqueeze(1)
                output = self.model(model_input)
                pred_scaled = output.item()
                predictions.append(pred_scaled)
                for _ in range(steps - 1):
                    model_input = torch.FloatTensor([[pred_scaled]]).unsqueeze(1)
                    output = self.model(model_input)
                    pred_scaled = output.item()
                    predictions.append(pred_scaled)
                return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        except Exception as e:
            logging.error(f"Error making GRU predictions: {e}")
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
            logging.info(f"GRU Warmup metrics - MSE: {self.metrics['mse']:.4f}, MAE: {self.metrics['mae']:.4f}")
        except Exception as e:
            logging.error(f"Error during GRU warmup: {e}")
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
            logging.error(f"Error in GRU forecast: {e}")
            return np.zeros(steps)
