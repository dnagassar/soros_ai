import unittest
import numpy as np
from modules.signal_aggregator import aggregate_signals

class TestSignalAggregator(unittest.TestCase):
    def setUp(self):
        # Use 500 samples for X_train and 50 for X_test
        self.X_train = np.random.rand(500, 10)
        self.y_train = np.random.rand(500)
        self.X_test = np.random.rand(50, 10)
        self.signal_ages = [1, 1, 5, 10, 2]

    def test_aggregate_signals_without_social(self):
        final_signal = aggregate_signals("News headline", 1, "AAPL", self.X_train, self.y_train, self.X_test, self.signal_ages)
        self.assertIsInstance(final_signal, float)

    def test_aggregate_signals_with_social(self):
        final_signal = aggregate_signals("News headline", 1, "AAPL", self.X_train, self.y_train, self.X_test, self.signal_ages, social_query="AAPL")
        self.assertIsInstance(final_signal, float)

if __name__ == '__main__':
    unittest.main()
