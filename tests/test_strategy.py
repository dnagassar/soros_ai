import unittest
import backtrader as bt
from modules.strategy import SentimentStrategy

class TestStrategy(unittest.TestCase):
    def test_strategy_initialization(self):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SentimentStrategy)
        # Provide a sample data feed; ensure this file exists or adjust to use a dummy feed.
        data = bt.feeds.YahooFinanceCSVData(dataname='data/historical_prices.csv')
        cerebro.adddata(data)
        cerebro.run()
        # If run completes without error, the test is considered passed.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
