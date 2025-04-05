import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import pandas as pd
import time

# Import your backend modules
from Data_handler import DataHandler
from Feature_engineering import FeatureEngineering
from model import Model
from Signal_Generator import SignalGenerator
from Visualiser import Visualiser
from Riskmanager import RiskManager
from Indicators import Indicators
import traceback    # Import traceback for error handling

class LiveTradingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📡 Live Trading Interface")
        self.running = False
        self.tickers = ["AAPL"]
        self.mode = tk.StringVar(value="Backtest")
        self.start_time = None
        self.current_balance = 0
        self.trades_made = 0
        self.wins = 0
        self.losses = 0

        self.setup_ui()
        self.ticker_selector.set("AAPL")

    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(expand=True, fill='both', padx=10, pady=10)

        # Left panel for setup
        left_panel = tk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill='y', padx=(0, 10))

        # Ticker controls
        ticker_frame = tk.LabelFrame(left_panel, text="Ticker Management")
        ticker_frame.pack(fill='x', pady=(0, 10))

        tk.Label(ticker_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
        self.ticker_entry = tk.Entry(ticker_frame, width=10)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(ticker_frame, text="Add", command=self.add_ticker).pack(side=tk.LEFT, padx=5)

        self.ticker_listbox = tk.Listbox(left_panel, height=5, selectmode=tk.SINGLE)
        self.ticker_listbox.pack(fill='x', pady=5)
        self.ticker_listbox.insert(tk.END, "AAPL")
        tk.Button(left_panel, text="Remove", command=self.remove_ticker).pack(fill='x')

        # Settings frame
        self.settings_frame = tk.LabelFrame(left_panel, text="Trading Settings")
        self.settings_frame.pack(fill='x', pady=10)

        self.balance_entry = tk.Entry(self.settings_frame, width=10)
        self.interval_var = tk.StringVar(value="1d")
        self.interval_menu = ttk.Combobox(self.settings_frame, textvariable=self.interval_var, values=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo'], state='readonly', width=7)
        self.start_date_entry = tk.Entry(self.settings_frame, width=12)
        self.end_date_entry = tk.Entry(self.settings_frame, width=12)

        self.live_start_label = tk.Label(self.settings_frame, text="Start Time:")
        self.live_elapsed_label = tk.Label(self.settings_frame, text="Elapsed Time:")
        self.live_start_value = tk.Label(self.settings_frame, text="N/A")
        self.live_elapsed_value = tk.Label(self.settings_frame, text="N/A")

        settings = [
            ("Starting Balance:", self.balance_entry, "10000"),
            ("Interval:", self.interval_menu, "1d"),
            ("Start Date:", self.start_date_entry, (datetime.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')),
            ("End Date:", self.end_date_entry, datetime.now().strftime('%Y-%m-%d'))
        ]

        self.setting_widgets = []
        for i, (label, entry, default) in enumerate(settings):
            lbl = tk.Label(self.settings_frame, text=label)
            lbl.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            entry.insert(0, default)
            self.setting_widgets.append((lbl, entry))

        tk.Label(self.settings_frame, text="Mode:").grid(row=len(settings), column=0, sticky='w', padx=5, pady=2)
        mode_menu = tk.OptionMenu(self.settings_frame, self.mode, "Backtest", "Live", command=self.toggle_mode_view)
        mode_menu.grid(row=len(settings), column=1, sticky='w', padx=5, pady=2)

        # Control buttons
        control_frame = tk.Frame(left_panel)
        control_frame.pack(fill='x', pady=10)

        self.start_button = tk.Button(control_frame, text="▶️ Start Trading", command=self.start_trading)
        self.start_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.stop_button = tk.Button(control_frame, text="⏹️ Stop", command=self.stop_trading)
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=2)

        # Center panel for plot and ticker selector
        center_panel = tk.Frame(main_container)
        center_panel.pack(side=tk.LEFT, expand=True, fill='both')

        selector_frame = tk.Frame(center_panel)
        selector_frame.pack(fill='x')
        tk.Label(selector_frame, text="📊 View Ticker:").pack(side=tk.LEFT, padx=5)
        self.ticker_selector = ttk.Combobox(selector_frame, values=self.tickers)
        self.ticker_selector.pack(side=tk.LEFT, padx=5)
        self.ticker_selector.bind("<<ComboboxSelected>>", lambda e: self.plot_data(None, self.ticker_selector.get()))

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=center_panel)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        # Right panel for statistics
        right_panel = tk.LabelFrame(main_container, text="Trading Statistics")
        right_panel.pack(side=tk.RIGHT, fill='y', padx=(10, 0))

        self.stats_labels = {"Overall": {}, "Current Ticker": {}}

        for section in ["Overall", "Current Ticker"]:
            section_frame = tk.LabelFrame(right_panel, text=section)
            section_frame.pack(fill='x', pady=5)
            stats = [
                "Ticker", "Mode", "Starting Balance", "Current Balance",
                "Trades Made", "Wins", "Losses", "Win Rate",
                "Current Position", "Last Signal", "Prediction"
            ]
            for stat in stats:
                frame = tk.Frame(section_frame)
                frame.pack(fill='x', padx=5, pady=2)
                tk.Label(frame, text=f"{stat}:", anchor='w').pack(side=tk.LEFT)
                self.stats_labels[section][stat] = tk.Label(frame, text="N/A", anchor='e')
                self.stats_labels[section][stat].pack(side=tk.RIGHT)

    def toggle_mode_view(self, mode):
        if mode == "Live":
            for lbl, entry in self.setting_widgets[2:]:
                lbl.grid_remove()
                entry.grid_remove()
            self.live_start_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
            self.live_start_value.grid(row=2, column=1, sticky='w')
            self.live_elapsed_label.grid(row=3, column=0, sticky='w', padx=5, pady=2)
            self.live_elapsed_value.grid(row=3, column=1, sticky='w')
        else:
            for lbl, entry in self.setting_widgets:
                lbl.grid()
                entry.grid()
            self.live_start_label.grid_remove()
            self.live_start_value.grid_remove()
            self.live_elapsed_label.grid_remove()
            self.live_elapsed_value.grid_remove()

    def add_ticker(self):
        ticker = self.ticker_entry.get().upper()
        if ticker and ticker not in self.tickers:
            self.tickers.append(ticker)
            self.ticker_listbox.insert(tk.END, ticker)
            self.ticker_selector.config(values=self.tickers)
            self.ticker_entry.delete(0, tk.END)

    def remove_ticker(self):
        selection = self.ticker_listbox.curselection()
        if selection:
            index = selection[0]
            ticker = self.ticker_listbox.get(index)
            self.ticker_listbox.delete(index)
            self.tickers.remove(ticker)
            self.ticker_selector.config(values=self.tickers)

    def start_trading(self):
        if not self.tickers:
            messagebox.showwarning("Missing Tickers", "Please add at least one ticker.")
            return

        self.running = True
        self.start_time = time.time()
        self.live_loop()

    def stop_trading(self):
        self.running = False

    def live_loop(self):
        if not self.running:
            return

        threading.Thread(target=self.run_pipeline, daemon=True).start()
        interval_str = self.interval_var.get()
        interval = 1000 if interval_str == '1m' else 5000  # Placeholder delay based on resolution
        self.root.after(interval, self.live_loop)

    def run_pipeline(self):
        try:
            ticker = self.tickers[0]  # For now, process one at a time
            self.ticker_selector.set(ticker)

            if self.mode.get() == "Live":
                start_date = (datetime.now() - pd.Timedelta(days=1))
                end_date = datetime.now()
                elapsed = int(time.time() - self.start_time)
                self.live_start_value.config(text=start_date.strftime("%Y-%m-%d %H:%M:%S"))
                self.live_elapsed_value.config(text=f"{elapsed} sec")
            else:
                start_date = pd.to_datetime(self.start_date_entry.get())
                end_date = pd.to_datetime(self.end_date_entry.get())

            initial_balance = float(self.balance_entry.get())
            interval_str = self.interval_var.get()

            data = DataHandler.load_from_yfinance(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval=interval_str)
            data = Indicators.relative_strength_index(data)
            data = Indicators.macd(data)
            data = Indicators.average_true_range(data)
            data = Indicators.bollinger_bands(data)
            data = Indicators.moving_average(data)
            data = Indicators.exponential_moving_average(data)

            data = FeatureEngineering.engineer_features(data)

            model, _ = Model.train_model(data)
            data = Model.apply_model(data, model)
            data = SignalGenerator.generate_rule_based_signals(data)
            data = SignalGenerator.generate_consensus_signal(data)
            data = SignalGenerator.backtest(data, initial_balance=initial_balance)

            self.current_balance = data['Balance'].iloc[-1] if 'Balance' in data.columns else initial_balance
            self.trades_made += 1
            self.wins += 1 if self.trades_made % 2 == 0 else 0
            self.losses += 1 if self.trades_made % 2 != 0 else 0

            self.plot_data(data, ticker)
            self.update_info_display(ticker, data)

        except Exception as e:
            print(f"Error in pipeline: {e}")
            traceback.print_exc()

    def plot_data(self, data, ticker):
        self.ax.clear()
        if data is None:
            return
        self.ax.plot(data['Date'], data['Close'], label='Close')

        if 'Consensus_Signal' in data.columns:
            buy = data[data['Consensus_Signal'] == 1]
            sell = data[data['Consensus_Signal'] == -1]
            self.ax.scatter(buy['Date'], buy['Close'], marker='^', color='g', label='Buy')
            self.ax.scatter(sell['Date'], sell['Close'], marker='v', color='r', label='Sell')

        self.ax.set_title(f"{self.mode.get()} Trading - {ticker}")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        self.canvas.draw()

    def update_info_display(self, ticker, data):
        win_rate = (self.wins / self.trades_made * 100) if self.trades_made > 0 else 0
        last_signal = data['Consensus_Signal'].iloc[-1] if 'Consensus_Signal' in data.columns else 'N/A'
        current_position = "Long" if last_signal == 1 else "Short" if last_signal == -1 else "None"

        current_stats = {
            "Ticker": ticker,
            "Mode": self.mode.get(),
            "Starting Balance": f"${self.balance_entry.get()}",
            "Current Balance": f"${self.current_balance:.2f}",
            "Trades Made": str(self.trades_made),
            "Wins": str(self.wins),
            "Losses": str(self.losses),
            "Win Rate": f"{win_rate:.1f}%",
            "Current Position": current_position,
            "Last Signal": str(last_signal),
            "Prediction": str(data['Prediction'].iloc[-1] if 'Prediction' in data.columns else 'N/A')
        }

        for key, label in self.stats_labels['Overall'].items():
            label.config(text=current_stats.get(key, "N/A"))
        for key, label in self.stats_labels['Current Ticker'].items():
            label.config(text=current_stats.get(key, "N/A"))

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveTradingGUI(root)
    root.mainloop()
