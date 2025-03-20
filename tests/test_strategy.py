import unittest
import backtrader as bt
from modules.strategy import SentimentStrategy

class TestStrategy(unittest.TestCase):
    def test_strategy_subclass(self):
        self.assertTrue(issubclass(SentimentStrategy, bt.Strategy))

if __name__ == '__main__':
    unittest.main()
