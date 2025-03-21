# main.py

import logging
from config import LOG_DIR
from gui.app import StockForecastApp
import tkinter as tk

if __name__ == "__main__":
    # Logging is configured in config.py and used throughout the modules.
    root = tk.Tk()
    app = StockForecastApp(root)
    root.mainloop()
