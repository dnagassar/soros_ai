# main.py
from modules.asset_selector import select_top_assets
from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.logger import log_trade
from modules.strategy import AdaptiveSentimentStrategy
import backtrader as bt

assets_to_trade = select_top_assets(n=10)

for symbol in assets_to_trade:
    data = fetch_price_data(symbol, '2023-01-01', '2023-12-31')
    data.to_csv(f'data/{symbol}_prices.csv')

    # Further processing and trading logic here...
    # e.g., run aggregation and trading strategy per asset
