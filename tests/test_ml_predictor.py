import unittest
import numpy as np
import pandas as pd
from modules.ml_predictor import ensemble_predict_wrapper
from modules.ml_pipeline import MLPipeline

class TestMLIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic price data
        dates = pd.date_range(start='2023-01-01', periods=200)
        self.price_data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 200),
            'High': np.random.normal(105, 5, 200),
            'Low': np.random.normal(95, 5, 200),
            'Close': np.random.normal(100, 5, 200),
            'Volume': np.random.normal(1000000, 100000, 200)
        }, index=dates)
        
        # Add some technical indicators
        self.price_data['MA10'] = self.price_data['Close'].rolling(window=10).mean()
        self.price_data['MA50'] = self.price_data['Close'].rolling(window=50).mean()
        self.price_data['RSI'] = 50 + np.random.normal(0, 10, 200)  # Simplified RSI
        
        # Drop NaNs
        self.price_data = self.price_data.dropna()
        
        # Train/test split
        self.X_train = self.price_data.iloc[:-30].copy()
        self.y_train = self.X_train['Close'].pct_change().shift(-1).iloc[:-1]
        self.X_test = self.price_data.iloc[-30:].copy()
    
    def test_ml_pipeline_feature_preparation(self):
        """Test if ML pipeline can prepare features properly"""
        try:
            ml_pipeline = MLPipeline()
            X, y, features = ml_pipeline.prepare_features(self.price_data, 'TEST')
            
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertIsNotNone(features)
            self.assertGreater(len(X), 0)
            self.assertGreater(len(y), 0)
            self.assertGreater(len(features), 0)
        except Exception as e:
            self.fail(f"Feature preparation failed: {str(e)}")
    
    def test_ensemble_predict_wrapper(self):
        """Test if ensemble predict wrapper works with DataFrame inputs"""
        try:
            predictions = ensemble_predict_wrapper(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test
            )
            
            self.assertIsNotNone(predictions)
            self.assertEqual(len(predictions), len(self.X_test))
        except Exception as e:
            self.fail(f"Ensemble prediction failed: {str(e)}")