import unittest
import pandas as pd
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data

class TestDataAcquisition(unittest.TestCase):
    
    def test_price_data_fetch_single(self):
        """Test if individual price data can be fetched"""
        try:
            data = fetch_price_data('AAPL', '2023-01-01', '2023-01-31')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertFalse(data.empty)
            self.assertTrue('Close' in data.columns)
            self.assertTrue('Open' in data.columns)
            self.assertTrue('High' in data.columns)
            self.assertTrue('Low' in data.columns)
            self.assertTrue('Volume' in data.columns)
        except Exception as e:
            self.fail(f"Failed to fetch price data: {str(e)}")
    
    def test_synthetic_data_generation(self):
        """Test if synthetic data is generated when real data is unavailable"""
        # Testing with a fake ticker should trigger synthetic data generation
        data = fetch_price_data('FAKESYMBOL123', '2023-01-01', '2023-01-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertTrue('Close' in data.columns)
    
    def test_batch_fetch(self):
        """Test batch fetching functionality"""
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            batch_data = batch_fetch_price_data(symbols, '2023-01-01', '2023-01-31')
            
            self.assertIsInstance(batch_data, dict)
            self.assertEqual(len(batch_data), len(symbols))
            
            for symbol in symbols:
                self.assertIn(symbol, batch_data)
                self.assertIsInstance(batch_data[symbol], pd.DataFrame)
                self.assertFalse(batch_data[symbol].empty)
        except Exception as e:
            self.fail(f"Failed to batch fetch price data: {str(e)}")