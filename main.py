# main.py
from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.logger import log_trade
from modules.strategy import AdaptiveSentimentStrategy
import backtrader as bt
import numpy as np

# --- Step 1: Data Acquisition ---
data = fetch_price_data('AAPL', '2020-01-01', '2020-12-31')
# Optionally, save CSV for reference:
data.to_csv('data/historical_prices.csv')

# --- Step 2: Signal Aggregation Example ---
news_text = "Company reports record earnings, bullish sentiment prevails."
technical_signal = 1   # Example technical indicator signal
symbol = "AAPL"
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_test = np.random.rand(1, 10)
signal_ages = [1, 1, 5, 10, 2]  # Example ages (in days) for each signal

final_signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test, signal_ages)
print("Final Aggregated Signal:", final_signal)
log_trade(f"Aggregated Signal for {symbol}: {final_signal}")

# --- Step 3: Execute Trading Strategy with Adaptive Position Sizing ---
cerebro = bt.Cerebro()

# Pass the aggregated signal as a parameter to the strategy
cerebro.addstrategy(AdaptiveSentimentStrategy, signal=final_signal)

# Use PandasData feed with the DataFrame (make sure columns are flattened)
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

cerebro.run()
cerebro.plot()
