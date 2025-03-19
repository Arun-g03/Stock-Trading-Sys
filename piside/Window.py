import sys
import os
import tempfile
import yfinance as yf
import plotly.graph_objects as go
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QComboBox, QStackedWidget, QLabel
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView  # Ensure correct import

class MainMenu(QWidget):
    """Main Menu Page - Select a stock and go to the plot page"""
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()

        # Dropdown for tickers
        self.ticker_dropdown = QComboBox()
        self.ticker_dropdown.addItems(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])  # Add more tickers as needed
        layout.addWidget(self.ticker_dropdown)

        # "Go" Button
        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.load_plot_page)
        layout.addWidget(self.go_button)

        self.setLayout(layout)

    def load_plot_page(self):
        """Switch to the plot page and fetch stock data"""
        selected_ticker = self.ticker_dropdown.currentText()
        self.stacked_widget.widget(1).plot_stock_data(selected_ticker)
        self.stacked_widget.setCurrentIndex(1)  # Switch to plot page


class PlotPage(QWidget):
    """Plot Page - Displays stock price data with an interactive Plotly chart"""
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        self.layout = QVBoxLayout()

        # Title Label
        self.label = QLabel("Stock Data")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        # WebView to display the interactive Plotly chart
        self.chart_view = QWebEngineView()
        self.layout.addWidget(self.chart_view)

        # "Back to Menu" Button
        self.back_button = QPushButton("Back to Menu")
        self.back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.layout.addWidget(self.back_button)

        self.setLayout(self.layout)

    def plot_stock_data(self, ticker):
        """Fetch and plot stock data using Plotly"""
        stock_data = yf.download(ticker, period="1mo")  # Fetch 1 month of stock data

        if not stock_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"],
                                     mode='lines', name=f"{ticker} Close Price"))

            fig.update_layout(
                title=f"{ticker} - Closing Prices",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                height=500
            )

            # Save the figure as an HTML file
            temp_file = os.path.join(tempfile.gettempdir(), "stock_plot.html")
            fig.write_html(temp_file)

            # Ensure the file is correctly written before loading
            if os.path.exists(temp_file):
                # Force refresh the view by setting an empty URL first
                self.chart_view.setUrl(QUrl("about:blank"))
                self.chart_view.setUrl(QUrl.fromLocalFile(temp_file))  # Correct file URL format
            else:
                self.label.setText("Error: Unable to load the plot.")

        else:
            self.label.setText("No Data Available")


class MainWindow(QMainWindow):
    """Main Application Window"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Viewer - Interactive Plotly")
        self.resize(800, 600)

        # Stack Widget for Navigation
        self.stacked_widget = QStackedWidget()
        self.main_menu = MainMenu(self.stacked_widget)
        self.plot_page = PlotPage(self.stacked_widget)

        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.plot_page)

        self.setCentralWidget(self.stacked_widget)


# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
