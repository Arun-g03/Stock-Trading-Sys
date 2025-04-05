# forecasts/forecast_selector.py

from .lstm_forecast import LSTMForecast
from .gru_forecast import GRUForecast
from .arima_forecast import ARIMAForecast
from .holtwinters_forecast import HoltWintersForecast
from .linear_forecast import LinearRegressionForecast

class ForecasterSelector:
    @staticmethod
    def get_forecaster(method="LSTM"):
        if method == "LSTM":
            return LSTMForecast()
        elif method == "GRU":
            return GRUForecast()
        elif method == "ARIMA":
            return ARIMAForecast()
        elif method == "HoltWinters":
            return HoltWintersForecast()
        elif method == "LinearRegression":
            return LinearRegressionForecast()
        else:
            traceback.print_exc()
            raise ValueError("Invalid forecasting method.")
