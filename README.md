OUTDATED - NO LONGER WORKING ON SYSTEM

# Stock Analysis and Signal Generating System
# WORK IN PROGRESS, Current phase - The Jigsaw Puzzle - Getting the pieces together
# Hard to find time to work on this
## Purpose
This system aims to analyse historical stock data, identify patterns and indicators, and generate trading signals based on rule-based strategies and machine-learning models. The system aims to assist traders and investors in making informed decisions by providing accurate and timely trading signals.

## Goal
### The primary goals of this system are:

  Loading and preprocessing historical stock data from various sources.

  To initialise and compute various technical indicators and chart patterns.

  To generate trading signals using both rule-based and machine learning-based approaches.

  To provide visualisation tools for historical data, indicators, patterns, and trading signals.

  To backtest the trading strategies and evaluate their performance.

![UML SYSTEM DIAGRAM](UML_Diagram.png)



## System Overview
### 1. Data Handling
The system can load historical stock data from multiple sources:

CSV files: Load data from local CSV files.
Yahoo Finance: Fetch historical data using the yfinance library.
Alpaca API: Fetch historical data using the Alpaca trading API.

### 2. Feature Engineering
The system computes various technical indicators and chart patterns, which are then used as features for generating trading signals:

Indicators: Moving Averages, RSI, MACD, Bollinger Bands, ATR, etc.
Patterns: Higher Highs and Lower Lows, Double Top, Head and Shoulders, Cup and Handle, etc.
### 3. Model Training
The system trains machine learning models to predict trading signals:

RandomForestClassifier: Used to train a model based on historical features and corresponding signals.

### 4. Signal Generation
The system generates trading signals using the following:

Rule-Based Methods: Simple conditions based on indicators and patterns.
Model-Based Methods: Predictions from the trained machine learning model.
Consensus Signals: A combination of rule-based and model-based signals.

### 5. Backtesting and Evaluation
The system backtests the generated trading signals on historical data to evaluate the performance of the trading strategies.

### 6. Visualisation
The system will provide visualisation tools to plot:

Historical stock price data.
Indicators and patterns.
Buy/Sell signals.
Backtesting results and balance over time.
