import unittest
from modules.data_acquisition import fetch_price_data

class TestDataAcquisition(unittest.TestCase):
    def test_fetch_data(self):
        data = fetch_price_data('AAPL', '2024-01-01', '2024-01-10')
        self.assertFalse(data.empty)

if __name__ == '__main__':
    unittest.main()
