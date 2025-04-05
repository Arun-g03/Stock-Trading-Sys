import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from Indicators import Indicators

# Fetch Apple stock data
symbol = "AAPL"
data = yf.download(symbol, period="6mo", interval="1d")

# Flatten MultiIndex DataFrame
data.columns = [col[0] for col in data.columns]  # Keep only first level (Price)

# Ensure 'Close' column is accessible
if "Close" not in data.columns:
    raise KeyError("The 'Close' column is missing after processing. Check column structure.")

print("\n📌 Fixed DataFrame Structure:")
print(data.head())  # Confirm the column names are properly set



# Apply all indicators
indicators = Indicators()
for indicator in [
    indicators.moving_average,
    indicators.exponential_moving_average,
    indicators.relative_strength_index,
    indicators.bollinger_bands,
    indicators.macd,
    indicators.average_true_range,
    indicators.stochastic_oscillator,
    indicators.commodity_channel_index,
    indicators.ichimoku_cloud,
    indicators.aroon,
    indicators.parabolic_sar,
    indicators.volume_weighted_average_price,
    indicators.on_balance_volume,
    indicators.money_flow_index,
    indicators.chaikin_money_flow,
    indicators.ease_of_movement,
    indicators.accumulation_distribution,
    indicators.ultimate_oscillator,
    indicators.z_score,
    indicators.pivot_points,
    indicators.detrended_price_oscillator,
    indicators.rate_of_change,
]:
    try:
        data = indicator(data)
    except Exception as e:
        print(f"Error applying {indicator.__name__}: {e}")

# Plot each indicator separately
indicators_to_plot = [
    ("Close", "Closing Price"),
    ("MA_20", "20-Day Moving Average"),
    ("EMA_20", "20-Day Exponential Moving Average"),
    ("RSI", "Relative Strength Index"),
    ("BB_High", "Bollinger Bands High"),
    ("BB_Low", "Bollinger Bands Low"),
    ("MACD", "MACD"),
    ("MACD_Signal", "MACD Signal Line"),
    ("ATR", "Average True Range"),
    ("Stoch_K", "Stochastic Oscillator %K"),
    ("CCI", "Commodity Channel Index"),
    ("Ichimoku_Conversion", "Ichimoku Conversion Line"),
    ("Aroon_Up", "Aroon Up"),
    ("Aroon_Down", "Aroon Down"),
    ("Parabolic_SAR", "Parabolic SAR"),
    ("VWAP", "Volume Weighted Average Price"),
    ("OBV", "On Balance Volume"),
    ("MFI", "Money Flow Index"),
    ("CMF", "Chaikin Money Flow"),
    ("EOM", "Ease of Movement"),
    ("ADI", "Accumulation/Distribution Index"),
    ("Ultimate_Oscillator", "Ultimate Oscillator"),
    ("Z_Score", "Z-Score"),
    ("Pivot", "Pivot Points"),
    ("DPO", "Detrended Price Oscillator"),
    ("ROC", "Rate of Change"),
]

for column, title in indicators_to_plot:
    if column in data.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data[column], label=title, color='b')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(title)
        plt.legend()
        plt.grid()
        plt.show()