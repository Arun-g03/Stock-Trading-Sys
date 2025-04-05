import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import pandas as pd

# Import your backend modules
from Data_handler import DataHandler
from Feature_engineering import FeatureEngineering
from model import Model
from Signal_Generator import SignalGenerator
from Visualiser import Visualiser
from Riskmanager import RiskManager

class LiveTradingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📡 Live Trading Interface")
        self.running = False
        self.tickers = []

        self.setup_ui()

    def setup_ui(self):
        # Top frame: ticker input and control buttons
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Ticker:").grid(row=0, column=0)
        self.ticker_entry = tk.Entry(top_frame, width=10)
        self.ticker_entry.grid(row=0, column=1)
        tk.Button(top_frame, text="Add", command=self.add_ticker).grid(row=0, column=2)

        self.ticker_listbox = tk.Listbox(top_frame, height=5, selectmode=tk.SINGLE)
        self.ticker_listbox.grid(row=1, column=0, columnspan=3, pady=5)
        tk.Button(top_frame, text="Remove", command=self.remove_ticker).grid(row=2, column=0, columnspan=3)

        # Settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(pady=10)

        tk.Label(settings_frame, text="Starting Balance:").grid(row=0, column=0)
        self.balance_entry = tk.Entry(settings_frame, width=10)
        self.balance_entry.insert(0, "10000")
        self.balance_entry.grid(row=0, column=1)

        tk.Label(settings_frame, text="Interval (sec):").grid(row=0, column=2)
        self.interval_entry = tk.Entry(settings_frame, width=5)
        self.interval_entry.insert(0, "60")
        self.interval_entry.grid(row=0, column=3)

        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.start_button = tk.Button(control_frame, text="▶️ Start Live Trading", command=self.start_trading)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="⏹️ Stop", command=self.stop_trading)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Plot area
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(padx=10, pady=10)

    def add_ticker(self):
        ticker = self.ticker_entry.get().upper()
        if ticker and ticker not in self.tickers:
            self.tickers.append(ticker)
            self.ticker_listbox.insert(tk.END, ticker)
            self.ticker_entry.delete(0, tk.END)

    def remove_ticker(self):
        selection = self.ticker_listbox.curselection()
        if selection:
            index = selection[0]
            ticker = self.ticker_listbox.get(index)
            self.ticker_listbox.delete(index)
            self.tickers.remove(ticker)

    def start_trading(self):
        if not self.tickers:
            messagebox.showwarning("Missing Tickers", "Please add at least one ticker.")
            return

        self.running = True
        self.live_loop()

    def stop_trading(self):
        self.running = False

    def live_loop(self):
        if not self.running:
            return

        threading.Thread(target=self.run_pipeline, daemon=True).start()
        interval = int(self.interval_entry.get()) * 1000
        self.root.after(interval, self.live_loop)

    def run_pipeline(self):
        try:
            ticker = self.tickers[0]  # For now, handle one ticker at a time
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=2)

            data = DataHandler.load_from_yfinance(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            data = FeatureEngineering.engineer_features(data)

            model, _ = Model.train_model(data)
            data = Model.apply_model(data, model)
            data = SignalGenerator.generate_rule_based_signals(data)
            data = SignalGenerator.generate_consensus_signal(data)
            data = SignalGenerator.backtest(data, initial_balance=float(self.balance_entry.get()))

            self.plot_data(data, ticker)

        except Exception as e:
            print(f"Error in pipeline: {e}")

    def plot_data(self, data, ticker):
        self.ax.clear()
        self.ax.plot(data['Date'], data['Close'], label='Close')

        if 'Consensus_Signal' in data.columns:
            buy = data[data['Consensus_Signal'] == 1]
            sell = data[data['Consensus_Signal'] == -1]
            self.ax.scatter(buy['Date'], buy['Close'], marker='^', color='g', label='Buy')
            self.ax.scatter(sell['Date'], sell['Close'], marker='v', color='r', label='Sell')

        self.ax.set_title(f"Live Trading - {ticker}")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveTradingGUI(root)
    root.mainloop()