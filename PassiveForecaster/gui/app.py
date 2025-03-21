# gui/app.py

import logging
import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pytz import timezone

from config import (
    TICKERS, FORECAST_HORIZON, WARMUP_PERIOD,
    INTERVAL_LOOKBACK_LIMITS, INTERVAL, ALPACA_KEY, ALPACA_SECRET
)
from data.fetch_data import fetch_data
from forecasts.forecast_selector import ForecasterSelector


class StockForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Forecasts")
        self.current_ticker_index = 0
        self.use_warmup = tk.BooleanVar(value=True)
        self.warmup_period = tk.IntVar(value=WARMUP_PERIOD)
        
        self.selected_algorithms = {
            "LSTM": tk.BooleanVar(value=True),
            "GRU": tk.BooleanVar(value=True),
            "ARIMA": tk.BooleanVar(value=True),
            "HoltWinters": tk.BooleanVar(value=True),
            "LinearRegression": tk.BooleanVar(value=True)
        }
        
        self.algorithm_colors = {
            "LSTM": "red",
            "GRU": "green",
            "ARIMA": "purple",
            "HoltWinters": "orange",
            "LinearRegression": "brown"
        }

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Navigation buttons
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X)
        self.prev_button = tk.Button(button_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = tk.Button(button_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)

        # Controls for warmup period
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        warmup_frame = tk.Frame(control_frame)
        warmup_frame.pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(warmup_frame, text="Use Warmup", variable=self.use_warmup, command=self.update_plot).pack(side=tk.LEFT)
        tk.Label(warmup_frame, text="Periods:").pack(side=tk.LEFT)
        warmup_entry = tk.Entry(warmup_frame, textvariable=self.warmup_period, width=5)
        warmup_entry.pack(side=tk.LEFT)
        tk.Button(warmup_frame, text="Apply", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # Algorithm selection checkboxes
        algo_frame = tk.Frame(root)
        algo_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(algo_frame, text="Select Algorithms:").pack(side=tk.LEFT, padx=5)
        for algo, var in self.selected_algorithms.items():
            cb = tk.Checkbutton(algo_frame, text=algo, variable=var,
                                command=self.update_plot, fg=self.algorithm_colors[algo])
            cb.pack(side=tk.LEFT, padx=5)
        
        # Metrics display
        metrics_frame = tk.Frame(root)
        metrics_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(metrics_frame, text="Performance Metrics:").pack(side=tk.LEFT, padx=5)
        self.metrics_vars = {}
        for algo in self.selected_algorithms.keys():
            self.metrics_vars[algo] = tk.StringVar(value=f"{algo}: N/A")
            metrics_label = tk.Label(metrics_frame, textvariable=self.metrics_vars[algo],
                                     fg=self.algorithm_colors[algo], anchor="w")
            metrics_label.pack(side=tk.TOP, padx=10, anchor="w")

        # Matplotlib figure and toolbar
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.update_plot()

    def on_closing(self):
        """Handle the window close event."""
        self.root.destroy()
        sys.exit()

    def show_previous(self):
        self.current_ticker_index = (self.current_ticker_index - 1) % len(TICKERS)
        self.update_plot()

    def show_next(self):
        self.current_ticker_index = (self.current_ticker_index + 1) % len(TICKERS)
        self.update_plot()

    def update_plot(self):
        ticker = TICKERS[self.current_ticker_index]

        # Fetch data over a defined lookback period
        lookback_days = INTERVAL_LOOKBACK_LIMITS.get(INTERVAL, 30)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)
        data = fetch_data(
            ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            alpaca_key=ALPACA_KEY,
            alpaca_secret=ALPACA_SECRET,
            paper=True
        )
        if data is None or data.empty:
            logging.error("No data found for ticker %s", ticker)
            return

        # Determine the correct date column and convert to timezone-aware datetimes
        date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce', utc=True)\
                           .dt.tz_convert('Europe/London')

        # Determine warmup period and display window
        warmup = self.warmup_period.get() if self.use_warmup.get() else 0
        if warmup > 0 and len(data) <= warmup:
            warmup = 0
            logging.warning("Not enough data for warmup. Disabling warmup.")
        display_points = max(warmup + FORECAST_HORIZON, 50)
        historical_data = data.iloc[-display_points:]

        # Select the closing price column (strip ticker prefix if needed)
        close_col = f'Close_{ticker}' if f'Close_{ticker}' in historical_data.columns else 'Close'
        if close_col not in historical_data.columns:
            logging.error("Column %s not found in data.", close_col)
            return

        # Clear the axes and set titles/labels
        self.ax.clear()
        self.ax.set_title(f'{ticker} Stock Price Forecast Comparison')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price')

        historical_x = historical_data[date_col].values
        historical_y = historical_data[close_col].values
        self.ax.plot(historical_x, historical_y, label='Historical Data', color='blue')

        # Plot current time line (using Europe/London time)
        current_time = pd.Timestamp.now(tz='Europe/London').replace(second=0, microsecond=0)
        self.ax.axvline(x=current_time, color='gray', linestyle='--', alpha=0.7, label='Current Time')

        # Define market open/close times in London time
        market_open = pd.to_datetime("14:30").time()  # 9:30am EST becomes 14:30 in GMT
        market_close = pd.to_datetime("21:00").time()   # 4:00pm EST becomes 21:00 in GMT

        # Shade off-market hours for each unique trading day
        unique_dates = pd.to_datetime(historical_data[date_col]).dt.date.unique()
        for d in unique_dates:
            day_start = pd.Timestamp.combine(d, pd.to_datetime("00:00").time()).tz_localize('Europe/London')
            day_open = pd.Timestamp.combine(d, market_open).tz_localize('Europe/London')
            day_close = pd.Timestamp.combine(d, market_close).tz_localize('Europe/London')
            day_end = pd.Timestamp.combine(d, pd.to_datetime("23:59").time()).tz_localize('Europe/London')
            self.ax.axvspan(day_start, day_open, facecolor='gray', alpha=0.08)
            self.ax.axvspan(day_close, day_end, facecolor='gray', alpha=0.08)

        # Determine time interval between data points in minutes
        if len(historical_x) >= 2:
            time_deltas = pd.Series(historical_x).diff().dropna()
            time_interval_minutes = int(time_deltas.mode()[0].total_seconds() // 60)
        else:
            time_interval_minutes = 60

        # Helper: Check if a datetime is during market hours (based on US/Eastern)
        def is_market_open(dt):
            dt_eastern = dt.astimezone(timezone('US/Eastern'))
            return (dt_eastern.weekday() < 5 and 
                    market_open <= dt_eastern.time() <= market_close)

        # Compute forecast timestamps ensuring they fall within market hours
        forecast_x = []
        dt = current_time.to_pydatetime().replace(second=0, microsecond=0)
        while len(forecast_x) < FORECAST_HORIZON:
            dt += timedelta(minutes=time_interval_minutes)
            if is_market_open(dt):
                forecast_x.append(np.datetime64(dt))

        # Mark the start of the warmup period if used
        if warmup > 0:
            warmup_start_time = historical_data[date_col].iloc[-warmup]
            self.ax.axvline(x=warmup_start_time, color='lightgray', linestyle=':', alpha=0.5, label='Warmup Start')

        # Remove ticker prefixes for forecasting models
        standard_data = data.copy()
        standard_data.columns = [col.replace(f"_{ticker}", "") for col in data.columns]

        # Plot forecasts and warmup predictions for each selected algorithm
        for algo_name, selected in self.selected_algorithms.items():
            if selected.get():
                try:
                    model = ForecasterSelector.get_forecaster(algo_name)
                    future_forecast = model.forecast(standard_data, steps=FORECAST_HORIZON, warmup=warmup)
                    
                    if warmup > 0 and model.warmup_predictions is not None and len(model.warmup_predictions) > 0:
                        warmup_x = historical_data[date_col].iloc[-warmup:].values
                        warmup_pred = model.warmup_predictions[:len(warmup_x)]
                        self.ax.plot(warmup_x, warmup_pred, linestyle='--',
                                     color=self.algorithm_colors[algo_name], alpha=0.7,
                                     label=f"Warmup {algo_name}")
                        if model.metrics['mse'] is not None and model.metrics['mae'] is not None:
                            metrics_text = f"{algo_name}: MSE: {model.metrics['mse']:.4f}, MAE: {model.metrics['mae']:.4f}"
                            self.metrics_vars[algo_name].set(metrics_text)
                    else:
                        self.metrics_vars[algo_name].set(f"{algo_name}: N/A")
                    
                    self.ax.plot(forecast_x, future_forecast, label=algo_name,
                                 color=self.algorithm_colors[algo_name])
                except Exception as e:
                    logging.error("Error plotting %s: %s", algo_name, e)
                    self.metrics_vars[algo_name].set(f"{algo_name}: Error")

        self.ax.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Enable hover tooltips on plotted lines
        def format_tooltip(sel):
            try:
                x_value = sel.target[0]
                if isinstance(x_value, (int, float)):
                    x_value = mdates.num2date(x_value)
                formatted_text = (
                    f"{sel.artist.get_label()}\n"
                    f"{pd.to_datetime(x_value).strftime('%Y-%m-%d %H:%M')}\n"
                    f"{sel.target[1]:.2f}"
                )
                sel.annotation.set_text(formatted_text)
            except Exception as e:
                logging.error("Tooltip Error: %s", e)

        cursor = mplcursors.cursor(self.ax.lines, hover=True)
        cursor.connect("add", format_tooltip)

        self.canvas.draw()
