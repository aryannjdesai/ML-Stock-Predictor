ML Stock Predictor

A beginner-friendly machine learning project that predicts stock price trends using historical financial data and technical indicators.

Project Overview

This project uses an LSTM neural network model built with PyTorch to predict whether a stock's closing price will go up or down the next day. It combines real-world financial indicators like:
- Moving Averages (MA5, MA10)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (Upper & Lower)
- Volatility and Returns

These indicators are treated as patterns that help the model make smarter predictions, which are then visualized in simple charts.

What You’ll Learn

- How to collect real stock market data with `yfinance`
- How to calculate common financial indicators using the `ta` library
- How to process time-series data for machine learning
- How to build and train an LSTM model with PyTorch
- How to evaluate predictions using accuracy and charts

Tech Stack

- Python
- PyTorch
- Pandas & NumPy
- yFinance & ta
- Matplotlib
- Scikit-learn

Project Structure
ML-Stock-Predictor/
│
├── main.py # Full ML pipeline (data prep, training, prediction, visualization)
├── stock_model_TSLA.pth # Saved PyTorch model
├── training_log_TSLA.txt # Training metrics
└── README.md # Project overview and instructions

Future Improvements

- Add multiple stock support with a dropdown selector
- Deploy as a web dashboard (e.g., Streamlit)
- Include sentiment analysis using financial news headlines
