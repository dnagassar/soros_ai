import unittest
from modules.data_acquisition import fetch_price_data

class TestDataAcquisition(unittest.TestCase):
    def test_fetch_price_data(self):
        # Test that data is returned and contains a common column such as "Close"
        df = fetch_price_data('AAPL', '2020-01-01', '2020-01-10')
        self.assertFalse(df.empty, "Data should not be empty.")
        self.assertIn('Close', df.columns, "Data should contain a 'Close' column.")

if __name__ == '__main__':
    unittest.main()
