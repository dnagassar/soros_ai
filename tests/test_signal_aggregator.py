import unittest
import numpy as np
from modules.signal_aggregator import aggregate_signals

class TestSignalAggregator(unittest.TestCase):
    def test_aggregate_signals(self):
        news_text = "Strong positive news."
        technical_signal = 1
        symbol = "AAPL"
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(1, 10)
        signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test)
        self.assertTrue(isinstance(signal, float))
        self.assertTrue(-1 <= signal <= 1)

if __name__ == '__main__':
    unittest.main()
