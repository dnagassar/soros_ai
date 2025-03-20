import unittest
import pandas as pd
from modules.data_acquisition import fetch_price_data

class TestDataAcquisition(unittest.TestCase):
    def test_fetch_price_data(self):
        df = fetch_price_data('AAPL', '2020-01-01', '2020-01-10')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn('Close', df.columns)

if __name__ == '__main__':
    unittest.main()
