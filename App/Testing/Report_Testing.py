import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ReportGenerator import ReportGenerator
# Sample test data
trade_data = [
    {"ticker": "AAPL", "decision": "BUY", "entry_price": 175.5, "stop_loss": 170.0, "take_profit": 185.0},
    {"ticker": "MSFT", "decision": "SELL", "entry_price": 310.2, "stop_loss": 320.0, "take_profit": 300.0}
]

risk_assessment = [
    {"ticker": "AAPL", "risk_score": 0.75, "position_size": 10, "max_drawdown_check": True},
    {"ticker": "MSFT", "risk_score": 0.60, "position_size": 8, "max_drawdown_check": True}
]

forecast_results = [
    {"ticker": "AAPL", "predicted_price": 180.0, "confidence": 0.85},
    {"ticker": "MSFT", "predicted_price": 305.0, "confidence": 0.80}
]

# Initialize Report Generator
report_gen = ReportGenerator(output_dir="test_reports")  # Store test reports in a separate folder

# Generate test report
report_file_path = report_gen.generate_report(trade_data, risk_assessment, forecast_results, filename="test_report.xlsx")

print(f"Test Report generated: {report_file_path}")