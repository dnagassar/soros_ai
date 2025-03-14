# main.py
from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.strategy import execute_trade
from modules.logger import log_event
import pandas as pd
import numpy as np

def main():
    symbol = 'AAPL'

    # Step 1: Fetch historical price data
    log_event("Fetching historical price data...")
    data = fetch_price_data(symbol, '2023-01-01', '2024-03-01')
    data.dropna(inplace=True)
    data.to_csv('data/historical_prices.csv', index=False)
    log_event("Historical price data fetched and saved.")

    # Step 2: Prepare real ML training and testing data
    log_event("Preparing ML datasets...")
    data['Future_Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = data[feature_columns].values
    y = data['Future_Close'].values

    split_index = int(len(X) * 0.8)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]

    latest_features = X_test[-1].reshape(1, -1)
    log_event("ML datasets prepared successfully.")

    # Step 3: Real-time news and technical indicators
    news_text = "Apple announces better-than-expected earnings, stock expected to rise."
    technical_signal = 1  # Replace later with real technical analysis signals

    log_event("Aggregating signals...")
    signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, latest_features)
    log_event(f"Aggregated Signal for {symbol}: {signal}")

    # Step 4: Execute trade based on aggregated signal
    trade_quantity = 10
    if signal > 0:
        log_event(f"Executing BUY trade for {symbol}.")
        execute_trade(symbol, trade_quantity, 'buy')
    else:
        log_event(f"Executing SELL trade for {symbol}.")
        execute_trade(symbol, trade_quantity, 'sell')

    log_event("Trade execution completed.")

if __name__ == "__main__":
    main()
