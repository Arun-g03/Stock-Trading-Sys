import sys
import pandas as pd
from App.Data_handler import DataHandler
from App.Validator import Validator
from App.Feature_engineering import FeatureEngineering
from App.model import Model
from App.Signal_Generator import SignalGenerator
from App.Screener import StockScreener
from App.Forecasting import ForecasterSelector
from App.Riskmanager import RiskManager
from App.ReportGenerator import ReportGenerator
from App.Visualiser import Visualiser
from App.Indicators import Indicators
from App.Logger import System_Log


def train_forecasters(data):
    """Train forecasting models on historical data."""
    logger.info("Training forecasting models...")
    forecasting_models = ["LSTM", "GRU", "ARIMA", "HoltWinters", "LinearRegression"]
    train_size = int(len(data) * 0.8)  # Use 80% for training

    for model_name in forecasting_models:
        try:
            logger.info(f"Training {model_name} model...")
            forecaster = ForecasterSelector.get_forecaster(model_name)

            if hasattr(forecaster, "train"):
                forecaster.train(data.iloc[:train_size])  # Train on 80% of data

            logger.info(f"{model_name} training completed.")
        except Exception as e:
            logger.error(f"Error training {model_name} forecaster: {e}")


def forecast_stocks(data):
    """Generate stock forecasts using trained models."""
    logger.info("Running forecasting models...")
    forecasting_models = ["LSTM", "GRU", "ARIMA", "HoltWinters", "LinearRegression"]
    forecasts = {}
    forecast_data = data.iloc[int(len(data) * 0.8):]  # Use last 20% for forecasting

    for model_name in forecasting_models:
        try:
            logger.info(f"Running {model_name} forecaster...")
            forecaster = ForecasterSelector.get_forecaster(model_name)
            forecasts[model_name] = forecaster.forecast(forecast_data, steps=10)

            logger.info(f"{model_name} forecasting completed.")
        except Exception as e:
            logger.error(f"Error running {model_name} forecaster: {e}")

    return forecasts


def full_trading_pipeline():
    """Execute the full automated trading system."""
    tickers = ["AAPL"]
    start_date, end_date = "2022-01-01", "2022-12-31"
    initial_capital = 10000

    # Initialize components
    data_handler = DataHandler()
    validator = Validator()
    feature_engineer = FeatureEngineering()
    model_handler = Model()
    signal_generator = SignalGenerator()
    visualiser = Visualiser()
    risk_manager = RiskManager(risk_per_trade=0.02, max_drawdown=0.1)
    screener = StockScreener(min_market_cap=1e10, min_volume=1e6, min_volatility=0.02)
    report_generator = ReportGenerator()
    logger = System_Log.setup_logger("main")

    logger.info("Fetching and validating data...")
    data = data_handler.load_from_yfinance(tickers, start_date, end_date)
    validator.validate_data_integrity(data)
    validator.validate_data_quality(data)

    logger.info("Calculating technical indicators...")
    data = Indicators.relative_strength_index(data)
    data = Indicators.macd(data)
    data = Indicators.average_true_range(data)
    data = Indicators.bollinger_bands(data)
    data = Indicators.moving_average(data)
    data = Indicators.exponential_moving_average(data)

    # Generate trading signals
    logger.info("Generating trading signals...")
    data = signal_generator.generate_rule_based_signals(data)

    # Apply feature engineering
    logger.info("Applying feature engineering...")
    data = feature_engineer.engineer_features(data)

    # Train and apply the trading model
    logger.info("Training trading model...")
    model, train_accuracy = model_handler.train_model(data)
    data = model_handler.apply_model(data, model)
    data = signal_generator.generate_consensus_signal(data)

    # Backtest strategy
    logger.info("Backtesting trading strategy...")
    backtested_data = signal_generator.backtest(data)
    rule_acc, model_acc = signal_generator.evaluate_signals(backtested_data)
    logger.info(f"Rule-based accuracy: {rule_acc:.2f}, Model accuracy: {model_acc:.2f}")

    # Train forecasters
    train_forecasters(data)

    # Generate forecasts
    forecasts = forecast_stocks(data)

    # Risk management assessment
    risk_assessment = {"status": "skipped", "reason": "Risk management not executed"}
    if 'Balance' in backtested_data.columns:
        latest_close = backtested_data['Close'].iloc[-1]
        latest_balance = backtested_data['Balance'].iloc[-1]
        latest_atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 1.5

        trade_data = {
            'ticker': tickers[0],
            'entry_price': latest_close,
            'strategy': 'percentage',
            'capital': initial_capital,
            'atr': latest_atr,
            'liquidity_data': {'volume': data['volume'].iloc[-1], 'spread': 0.01}
        }

        risk_assessment = risk_manager.run(trade_data, portfolio_value=latest_balance)
    
    # Generate reports
    logger.info("Generating execution report...")
    report_file = report_generator.generate_report(
        backtested_data.to_dict(orient='records')[:10],
        [risk_assessment],
        {model: forecast.tolist() if hasattr(forecast, "tolist") else list(forecast) for model, forecast in forecasts.items()}
    )
    logger.info(f"Report saved at: {report_file}")

    # Plot results
    visualiser.plot_signals(backtested_data, tickers)
    visualiser.plot_balance(backtested_data, tickers)
    Visualiser.plot_forecasts(data, forecasts, tickers[0])

    logger.info("Trading pipeline completed successfully!")


def real_time_simulation(ticker="AAPL"):
    """Simulate real-time trading with the last week of minute data."""
    logger = System_Log.setup_logger("real_time")
    logger.info("Starting real-time simulation...")

    # 🔹 Load last week of minute-level data
    data_handler = DataHandler()
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"Fetching minute-level data for {ticker} from {start_date} to {end_date}...")
    data = data_handler.load_from_yfinance([ticker], start_date, end_date, interval="1m")

    if data.empty:
        logger.error("No data retrieved. Check if the market is closed or data source is down.")
        return

    # 🔹 Initialize components
    validator = Validator()
    feature_engineer = FeatureEngineering()
    model_handler = Model()
    signal_generator = SignalGenerator()
    risk_manager = RiskManager(risk_per_trade=0.02, max_drawdown=0.1)
    screener = StockScreener(min_market_cap=1e10, min_volume=1e6, min_volatility=0.02)
    report_generator = ReportGenerator()
    visualiser = Visualiser()

    # Step through the data minute by minute
    for i in range(len(data)):
        logger.info(f"Processing data at {data.index[i]}...")

        # Get the current slice of data up to the current minute
        live_data = data.iloc[: i + 1].copy()

        # Validate Data
        validator.validate_data_integrity(live_data)
        validator.validate_data_quality(live_data)

        # Calculate Technical Indicators
        live_data = Indicators.relative_strength_index(live_data)
        live_data = Indicators.macd(live_data)
        live_data = Indicators.average_true_range(live_data)
        live_data = Indicators.bollinger_bands(live_data)
        live_data = Indicators.moving_average(live_data)
        live_data = Indicators.exponential_moving_average(live_data)

        # Generate Trading Signals
        live_data = signal_generator.generate_rule_based_signals(live_data)

        # Apply Feature Engineering
        live_data = feature_engineer.engineer_features(live_data)

        # Train and Apply Trading Model
        model, _ = model_handler.train_model(live_data)
        live_data = model_handler.apply_model(live_data, model)
        live_data = signal_generator.generate_consensus_signal(live_data)

        # Risk Management
        if "Balance" in live_data.columns:
            latest_close = live_data["Close"].iloc[-1]
            latest_balance = live_data["Balance"].iloc[-1]
            latest_atr = live_data["ATR"].iloc[-1] if "ATR" in live_data.columns else 1.5

            trade_data = {
                "ticker": ticker,
                "entry_price": latest_close,
                "strategy": "percentage",
                "capital": 10000,
                "atr": latest_atr,
                "liquidity_data": {"volume": live_data["Volume"].iloc[-1], "spread": 0.01},
            }

            risk_assessment = risk_manager.run(trade_data, portfolio_value=latest_balance)
        else:
            risk_assessment = {"status": "skipped", "reason": "Balance data missing"}

        # Forecasting
        forecasts = {}
        forecasting_models = ["LSTM", "GRU", "ARIMA", "HoltWinters", "LinearRegression"]
        for model_name in forecasting_models:
            try:
                forecaster = ForecasterSelector.get_forecaster(model_name)
                forecasts[model_name] = forecaster.forecast(live_data, steps=10)
            except Exception as e:
                logger.error(f"Error running {model_name} forecaster: {e}")

        # Generate Report
        report_generator.generate_report(
            live_data.to_dict(orient="records")[-10:],
            [risk_assessment],
            {model: forecast.tolist() if hasattr(forecast, "tolist") else list(forecast) for model, forecast in forecasts.items()},
        )

        # Visualization
        visualiser.plot_signals(live_data, [ticker])
        visualiser.plot_balance(live_data, [ticker])
        Visualiser.plot_forecasts(live_data, forecasts, ticker)

        # Simulate real-time delay (optional, adjust for testing speed)
        time.sleep(1)  # Wait 1 second before processing the next minute

    logger.info("Real-time simulation completed!")

def main():
    """User menu for different execution modes."""
    while True:
        print("\n📈 Trading System Menu 📈")
        print("1️⃣ - Run Full Trading Pipeline")
        print("2️⃣ - Train Forecasting Models")
        print("3️⃣ - Run Forecasting Only")
        print("4️⃣ - Simulate Real-Time Trading")
        print("5️⃣ - Exit")

        choice = input("Select an option: ")

        if choice == "1":
            full_trading_pipeline()
        elif choice == "2":
            tickers = ["AAPL"]
            start_date, end_date = "2022-01-01", "2022-12-31"
            data_handler = DataHandler()
            data = data_handler.load_from_yfinance(tickers, start_date, end_date)
            train_forecasters(data)
        elif choice == "3":
            tickers = ["AAPL"]
            start_date, end_date = "2022-01-01", "2022-12-31"
            data_handler = DataHandler()
            data = data_handler.load_from_yfinance(tickers, start_date, end_date)
            forecast_stocks(data)
        elif choice == "4":
            real_time_simulation()
        elif choice == "5":
            print("Exiting trading system...")
            sys.exit()
        else:
            print("Invalid option. Please select 1, 2, 3, 4, or 5.")

if __name__ == '__main__':
    main()
