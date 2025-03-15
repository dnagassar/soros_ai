from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.strategy import execute_trade
from modules.logger import log_event
import numpy as np
import pandas as pd

def main():
    symbol = "BTC-USD"  # Example cryptocurrency ticker (Bitcoin)

    # Fetch historical crypto price data
    data = fetch_price_data(symbol, '2024-01-01', '2024-12-31')
    data.to_csv('data/historical_prices.csv', index=False)

    # Use realistic input data for ML prediction
    news_text = "Crypto market shows bullish momentum."
    technical_signal = 1  # Replace with real crypto technical analysis
    trade_quantity = 0.001  # Adjust based on your crypto trading budget

    X_train = data[['Open', 'High', 'Low', 'Close', 'Volume']].values[:-1]
    y_train = data['Close'].shift(-1).dropna().values
    latest_features = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:].values

    # Generate trading signal
    signal = aggregate_signals(news_text="Cryptocurrency market shows bullish momentum",
                               technical_signal=technical_signal,
                               symbol=symbol,
                               X_train=X_train,
                               y_train=y_train,
                               X_test=latest_features)

    print("Final Aggregated Signal:", signal)
    log_event(f"Aggregated Signal for {symbol}: {signal}")

    # Execute trade based on signal
    side = 'buy' if signal > 0 else 'sell'
    execute_trade(symbol.replace("-USD", "USD"), trade_quantity, side)

if __name__ == "__main__":
    main()
