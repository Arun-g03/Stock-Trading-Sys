from App.Data_handler import DataHandler
from App.Validator import Validator
from App.Feature_engineering import FeatureEngineering
from App.model import Model
from App.Signal_Generator import SignalGenerator
from App.Screener import StockScreener
from App.Forecasting import ForecastingFactory
from App.Riskmanager import RiskManager
from App.ReportGenerator import ReportGenerator
from App.Visualiser import Visualiser
from App.Indicators import Indicators
from App.Logger import System_Log


def main():
    # 🔹 Configuration
    tickers = ["AAPL"]
    start_date, end_date = "2022-01-01", "2022-12-31"
    initial_capital = 10000

    # 🔹 Initialize components
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

    logger.info("Starting trading pipeline...")

    # 🔹 Data Loading and Validation
    logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = data_handler.load_from_yfinance(tickers, start_date, end_date)
    validator.validate_data_integrity(data)
    validator.validate_data_quality(data)

    # 🔹 Step 1: Calculate Technical Indicators Before Signal Generation
    logger.info("Calculating technical indicators...")
    data = Indicators.relative_strength_index(data)  # Ensure RSI is added
    data = Indicators.macd(data)  # Ensure MACD is added
    data = Indicators.average_true_range(data)
    data = Indicators.bollinger_bands(data)
    data = Indicators.moving_average(data)
    data = Indicators.exponential_moving_average(data)

    # 🔹 Step 2: Verify All Required Indicators Exist Before Proceeding
    # Step 2: Verify All Required Indicators Exist Before Proceeding
    existing_columns = set(data.columns)

    # Find the correct RSI column dynamically
    rsi_column = next((col for col in existing_columns if col.startswith("RSI")), None)

    # Adjust required indicators dynamically
    required_indicators = [rsi_column, "MACD", "MACD_Signal"]
    missing_indicators = [col for col in required_indicators if col not in existing_columns or col is None]

    if missing_indicators:
        logger.error(f"Missing indicators: {missing_indicators}")
        print("Existing Columns in Data:", data.columns)  # Debugging
        raise ValueError(f"Missing indicators: {missing_indicators}")



    # **Step 3: Generate Trading Signals**
    logger.info("Generating rule-based signals...")
    data = signal_generator.generate_rule_based_signals(data)
    logger.info(f"Columns after signal generation: {list(data.columns)}")

    if 'Signal' not in data.columns:
        raise ValueError("Signal column is missing. Check Signal_Generator.py!")

    # 🔹 Feature Engineering (AFTER Signal Generation)
    logger.info("Applying feature engineering (creating lagged features)...")
    data = feature_engineer.engineer_features(data)

    # 🔹 Model Training & Prediction
    logger.info("Training model on engineered data...")
    model, train_accuracy = model_handler.train_model(data)
    data = model_handler.apply_model(data, model)

    data = signal_generator.generate_consensus_signal(data)
    backtested_data = signal_generator.backtest(data)
    rule_acc, model_acc = signal_generator.evaluate_signals(backtested_data)

    logger.info(f"Rule-based accuracy: {rule_acc:.2f}, Model accuracy: {model_acc:.2f}")

    # 🔹 Visualization
    visualiser.plot_signals(backtested_data, tickers)
    visualiser.plot_balance(backtested_data, tickers)

    # 🔹 Forecasting
    logger.info("Running forecasting model (ARIMA)...")
    forecaster = ForecastingFactory.get_forecaster("ARIMA", order=(5, 1, 0))
    forecast = forecaster.forecast(data, steps=10)

    # 🔹 Risk Management
    latest_close = backtested_data['Close'].iloc[-1]
    latest_balance = backtested_data['Balance'].iloc[-1]
    latest_atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 1.5

    trade_data = {
        'ticker': tickers,
        'entry_price': latest_close,
        'strategy': 'percentage',
        'capital': initial_capital,
        'atr': latest_atr,
        'liquidity_data': {'volume': data['Volume'].iloc[-1], 'spread': 0.01}
    }

    risk_assessment = risk_manager.run(trade_data, portfolio_value=latest_balance)

    # 🔹 Stock Screening
    data = FeatureEngineering.calculate_resistance(data)
    data = FeatureEngineering.calculate_support(data)
    data = FeatureEngineering.calculate_volatility(data)
    data = FeatureEngineering.calculate_volume_moving_average(data)


    screening_results = screener.run(data, tickers)
    logger.info(f"Stock screening results: {screening_results}")

    # 🔹 Report Generation
    logger.info("Generating trade execution report...")
    report_file = report_generator.generate_report(
        backtested_data.to_dict(orient='records')[:10],
        [risk_assessment],
        forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
    )

    logger.info(f"Report saved at: {report_file}")
    logger.info("Trading pipeline completed successfully!")


if __name__ == '__main__':
    main()
