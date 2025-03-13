# tests/test_strategy.py
import unittest
import os
from modules.data_acquisition import fetch_price_data
import backtrader as bt
from modules.strategy import SentimentStrategy

class TestStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the 'data' folder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        # Fetch sample data and save to CSV
        df = fetch_price_data('AAPL', '2020-01-01', '2020-01-10')
        df.to_csv('data/historical_prices.csv')
        print("Sample CSV data generated for tests.")

    def test_strategy_initialization(self):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SentimentStrategy)
        data_feed = bt.feeds.YahooFinanceCSVData(dataname='data/historical_prices.csv')
        cerebro.adddata(data_feed)
        cerebro.run()
        # If the strategy runs without error, the test passes.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
