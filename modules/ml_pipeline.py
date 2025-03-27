# modules/ml_pipeline.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def prepare_features(self, df, symbol):
        """Enhanced feature engineering with length preservation"""
        try:
            df = df.copy().ffill().fillna(0)
            
            # Basic features
            df['return_1d'] = df['Close'].pct_change()
            df['volatility'] = df['return_1d'].rolling(20).std().ffill()
            
            # Technical indicators
            for window in [5, 10, 20]:
                df[f'ma_{window}'] = df['Close'].rolling(window).mean().ffill()
                df[f'vol_{window}'] = df['return_1d'].rolling(window).std().ffill()
                
            # Target creation
            df['target'] = df['Close'].pct_change().shift(-1)
            df = df.dropna(subset=['target'])
            
            return (
                df.drop(columns=['target', 'Close']),
                df['target'],
                df.drop(columns=['target']).columns.tolist()
            )
        except Exception as e:
            logger.error(f"Feature prep failed for {symbol}: {e}")
            return None, None, None

    def train_model(self, X, y, symbol):
        """Enhanced with input validation"""
        try:
            if len(X) != len(y):
                raise ValueError(f"Length mismatch: X({len(X)}) != y({len(y)})")
                
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            
            # Save model with features
            model_path = os.path.join(self.model_dir, f"{symbol}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': X.columns.tolist(),
                    'trained_at': datetime.now()
                }, f)
                
            return model
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return None

    def predict(self, X, symbol):
        """Enhanced prediction with length checks"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}.pkl")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Ensure feature compatibility
            missing = set(model_data['features']) - set(X.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
                for f in missing:
                    X[f] = 0
                    
            predictions = model_data['model'].predict(X[model_data['features']])
            
            # Ensure length matches
            if len(predictions) != len(X):
                logger.warning(f"Adjusting predictions length to match input")
                predictions = predictions[:len(X)]
                
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return np.zeros(len(X)) if hasattr(X, '__len__') else np.array([0.0])