import unittest
from modules.signal_aggregator import aggregate_signals
import pandas as pd
import numpy as np

class TestSignalAggregation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Synthetic data
        self.X_train = pd.DataFrame(np.random.random((100, 5)), 
                                    columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y_train = pd.Series(np.random.random(100))
        self.X_test = pd.DataFrame(np.random.random((10, 5)), 
                                   columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    
    def test_signal_aggregation(self):
        """Test if signals can be aggregated correctly"""
        try:
            # The function signature suggests these are the required parameters
            # You might need to adjust based on the actual implementation
            signal = aggregate_signals(
                "Test headline",
                1,  # Example news_sentiment
                "AAPL",  # Symbol
                self.X_train,
                self.y_train,
                self.X_test,
                [1, 1, 5, 10, 2]  # Signal ages
            )
            
            self.assertIsNotNone(signal)
            self.assertIsInstance(signal, (int, float))
            self.assertTrue(-1 <= signal <= 1, "Signal should be between -1 and 1")
        except Exception as e:
            self.fail(f"Signal aggregation failed: {str(e)}")
    
    def test_signal_aggregation_with_empty_data(self):
        """Test signal aggregation with empty data"""
        try:
            # Test with empty DataFrames
            signal = aggregate_signals(
                "Test headline",
                1,
                "AAPL",
                pd.DataFrame(),
                pd.Series(),
                pd.DataFrame(),
                [1, 1, 5, 10, 2]
            )
            
            # Should return a default value or handle empty data gracefully
            self.assertIsNotNone(signal)
        except Exception as e:
            self.fail(f"Signal aggregation with empty data failed: {str(e)}")