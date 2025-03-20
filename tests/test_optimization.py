import unittest
import os
import pandas as pd
from modules.optimization import optimize_autogluon

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Create data directory if it doesn't exist
        if not os.path.exists("../data"):
            os.makedirs("../data")
        dummy_file = "../data/historical_prices.csv"
        if not os.path.isfile(dummy_file):
            df = pd.DataFrame({'Close': [100 + i for i in range(500)]})
            df.to_csv(dummy_file, index=False)

    def test_optimize_autogluon(self):
        best_params, best_rmse = optimize_autogluon(n_trials=1)
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_rmse, float)

if __name__ == '__main__':
    unittest.main()
