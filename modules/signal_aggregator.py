# modules/signal_aggregator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import pickle
import os
import hashlib
from modules.ml_predictor import ensemble_predict

logger = logging.getLogger(__name__)

class SignalAggregator:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize weights and performance tracking
        self.signal_weights = {
            'sentiment': 0.20, 'technical': 0.35, 
            'ml': 0.25, 'macro': 0.15, 'earnings': 0.05
        }
        self.signal_performance = {k: {'correct': 0, 'total': 0} for k in self.signal_weights}

    def get_ml_signal(self, symbol, X_train, y_train, X_test, force_refresh=False):
        """Enhanced with robust length validation"""
        try:
            # Validate input shapes
            if X_train is None or y_train is None or X_test is None:
                return 0.0
                
            if len(X_train) != len(y_train):
                logger.error(f"Train length mismatch: X={len(X_train)} y={len(y_train)}")
                return 0.0
                
            # Ensure consistent feature count
            if X_train.shape[1] != X_test.shape[1]:
                logger.error(f"Feature count mismatch: train={X_train.shape[1]} test={X_test.shape[1]}")
                return 0.0
                
            # Get predictions with length validation
            predictions = ensemble_predict(X_train, y_train, X_test)
            
            if predictions is None or len(predictions) != len(X_test):
                logger.error(f"Prediction length mismatch for {symbol}")
                return 0.0
                
            # Calculate signal with volatility scaling
            pred_mean = np.mean(predictions)
            volatility = np.std(predictions)
            scale_factor = min(50, 0.5/max(volatility, 0.01))
            signal = np.clip(pred_mean * scale_factor, -1, 1)
            
            return float(signal)
            
        except Exception as e:
            logger.error(f"ML signal error for {symbol}: {str(e)[:200]}")
            return 0.0

    def aggregate_signals(self, symbol, price_data=None, **kwargs):
        """Main aggregation with enhanced validation"""
        signals = {
            'technical': self.get_technical_signal(price_data),
            'ml': self.get_ml_signal(
                symbol,
                kwargs.get('X_train'),
                kwargs.get('y_train'), 
                kwargs.get('X_test')
            ),
            # ... other signals ...
        }
        
        # Validate all signals before aggregation
        valid_signals = {
            k: v for k, v in signals.items() 
            if v is not None and not np.isnan(v)
        }
        
        if not valid_signals:
            return {'signal': 0.0, 'confidence': 0}
            
        # Weighted aggregation
        weighted_sum = sum(
            s * self.signal_weights.get(k, 0) 
            for k, s in valid_signals.items()
        )
        
        return {
            'signal': np.clip(weighted_sum, -1, 1),
            'confidence': len(valid_signals)/len(signals),
            'components': valid_signals
        }