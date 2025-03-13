# main.py
from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.logger import log_trade
from modules.strategy import SentimentStrategy
import backtrader as bt
import numpy as np

# 1. Data Acquisition
data = fetch_price_data('AAPL', '2020-01-01', '2020-12-31')
data.to_csv('data/historical_prices.csv')

# 2. Prepare dummy inputs for signal aggregation
news_text = "Company reports record earnings, bullish sentiment prevails."
technical_signal = 1
symbol = "AAPL"
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_test = np.random.rand(1, 10)

# 3. Aggregate Signals
final_signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test)
print("Final Aggregated Signal:", final_signal)
log_trade(f"Aggregated Signal for {symbol}: {final_signal}")

# 4. Execute Trading Strategy using Backtrader
cerebro = bt.Cerebro()
cerebro.addstrategy(SentimentStrategy)
data_feed = bt.feeds.YahooFinanceCSVData(dataname='data/historical_prices.csv')
cerebro.adddata(data_feed)
cerebro.run()
cerebro.plot()
