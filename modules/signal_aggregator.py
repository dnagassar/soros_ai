# modules/signal_aggregator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import pickle
import os
import hashlib
from modules.ml_predictor import ensemble_predict
from modules.timeseries_forecaster import TimeSeriesForecaster

logger = logging.getLogger(__name__)

class SignalAggregator:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize time series forecaster
        try:
            self.ts_forecaster = TimeSeriesForecaster(model_type='nbeats')
            self.ts_available = True
        except ImportError:
            logger.warning("TimeSeries forecaster not available - required libraries missing")
            self.ts_available = False
        
        # Initialize weights and performance tracking
        # Updated to include timeseries component
        self.signal_weights = {
            'sentiment': 0.15, 'technical': 0.30, 
            'ml': 0.20, 'timeseries': 0.20, 
            'macro': 0.10, 'earnings': 0.05
        }
        self.signal_performance = {k: {'correct': 0, 'total': 0} for k in self.signal_weights}

    def get_technical_signal(self, price_data):
        """Calculate technical indicators signal"""
        if price_data is None or len(price_data) < 50:
            return 0.0
        
        try:
            signals = []
            
            # Moving Average Crossover
            ma_short = price_data['Close'].rolling(window=20).mean()
            ma_long = price_data['Close'].rolling(window=50).mean()
            
            if ma_short.iloc[-1] > ma_long.iloc[-1]:
                # Bullish - scale by distance between MAs
                ma_signal = min((ma_short.iloc[-1] / ma_long.iloc[-1] - 1) * 10, 1)
            else:
                # Bearish - scale by distance between MAs
                ma_signal = max((ma_short.iloc[-1] / ma_long.iloc[-1] - 1) * 10, -1)
            
            signals.append(ma_signal)
            
            # RSI
            delta = price_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rs = rs.fillna(0)
            rsi = 100 - (100 / (1 + rs))
            
            # Convert RSI to signal (-1 to 1)
            if rsi.iloc[-1] > 70:
                rsi_signal = -((rsi.iloc[-1] - 70) / 30)  # Overbought - negative signal
            elif rsi.iloc[-1] < 30:
                rsi_signal = (30 - rsi.iloc[-1]) / 30  # Oversold - positive signal
            else:
                rsi_signal = 0  # Neutral
            
            signals.append(rsi_signal)
            
            # Bollinger Bands
            window = 20
            ma = price_data['Close'].rolling(window=window).mean()
            std = price_data['Close'].rolling(window=window).std()
            upper_band = ma + (std * 2)
            lower_band = ma - (std * 2)
            
            close = price_data['Close'].iloc[-1]
            
            if close > upper_band.iloc[-1]:
                bb_signal = -1  # Overbought
            elif close < lower_band.iloc[-1]:
                bb_signal = 1  # Oversold
            else:
                # Position within the bands
                width = upper_band.iloc[-1] - lower_band.iloc[-1]
                if width > 0:
                    position = (close - lower_band.iloc[-1]) / width
                    bb_signal = 1 - 2 * position  # 1 at lower band, -1 at upper band
                else:
                    bb_signal = 0
            
            signals.append(bb_signal)
            
            # MACD
            ema12 = price_data['Close'].ewm(span=12).mean()
            ema26 = price_data['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            
            # MACD Signal
            if macd.iloc[-1] > signal_line.iloc[-1]:
                macd_signal = min((macd.iloc[-1] / signal_line.iloc[-1] - 1) * 5, 1)
            else:
                macd_signal = max((macd.iloc[-1] / signal_line.iloc[-1] - 1) * 5, -1)
            
            signals.append(macd_signal)
            
            # Volume analysis
            if 'Volume' in price_data.columns:
                avg_volume = price_data['Volume'].rolling(window=20).mean()
                vol_ratio = price_data['Volume'].iloc[-1] / avg_volume.iloc[-1]
                
                # Volume signal - weight by price direction
                price_direction = np.sign(price_data['Close'].iloc[-1] - price_data['Close'].iloc[-2])
                volume_signal = price_direction * min(max(vol_ratio - 1, -1), 1) * 0.5
                
                signals.append(volume_signal)
            
            # Calculate average signal
            if signals:
                return np.clip(sum(signals) / len(signals), -1, 1)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Technical signal error: {str(e)[:200]}")
            return 0.0

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

    def get_timeseries_signal(self, symbol, price_data, force_refresh=False):
        """Get time series forecast signal for a symbol"""
        # Skip if time series forecaster not available
        if not self.ts_available:
            return 0.0
            
        try:
            # Only use recent data (last 6 months)
            if price_data is None or len(price_data) < 30:
                return 0.0
                
            recent_data = price_data.tail(126)  # ~6 months of trading days
            
            # Add basic technical features
            feature_cols = []
            for window in [5, 10, 20]:
                recent_data[f'ma_{window}'] = recent_data['Close'].rolling(window=window).mean()
                recent_data[f'vol_{window}'] = recent_data['Close'].pct_change().rolling(window=window).std()
                feature_cols.extend([f'ma_{window}', f'vol_{window}'])
            
            # Fill missing values
            recent_data = recent_data.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate signal
            signal = self.ts_forecaster.calculate_signal(
                symbol=symbol,
                df=recent_data,
                target_col='Close',
                feature_cols=feature_cols,
                n_forecast=5
            )
            
            return float(signal)
            
        except Exception as e:
            logger.error(f"Time series signal error for {symbol}: {str(e)[:200]}")
            return 0.0
    
    def get_sentiment_signal(self, symbol, **kwargs):
        """Get sentiment signal (placeholder - implement with your sentiment analysis code)"""
        # This is a placeholder for the sentiment signal
        # In a real implementation, this would call your sentiment analysis code
        return 0.0
    
    def get_macro_signal(self, symbol, **kwargs):
        """Get macroeconomic signal (placeholder - implement with your macro analysis code)"""
        # This is a placeholder for the macro signal
        # In a real implementation, this would call your macro analysis code
        return 0.0
    
    def get_earnings_signal(self, symbol, **kwargs):
        """Get earnings signal (placeholder - implement with your earnings analysis code)"""
        # This is a placeholder for the earnings signal
        # In a real implementation, this would call your earnings analysis code
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
            'timeseries': self.get_timeseries_signal(symbol, price_data),
            'sentiment': self.get_sentiment_signal(symbol, **kwargs),
            'macro': self.get_macro_signal(symbol, **kwargs),
            'earnings': self.get_earnings_signal(symbol, **kwargs)
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
        
    def track_signal_performance(self, signals, actual_return):
        """Track signal performance for adaptive weighting"""
        # Convert actual return to binary signal (-1, 0, 1)
        if actual_return > 0.01:  # 1% threshold for positive
            actual = 1
        elif actual_return < -0.01:  # -1% threshold for negative
            actual = -1
        else:
            actual = 0
            
        # Update performance metrics for each signal type
        for signal_type, value in signals.items():
            if signal_type in self.signal_performance:
                # Consider signal correct if it correctly predicted direction
                predicted = 1 if value > 0.2 else (-1 if value < -0.2 else 0)
                correct = (predicted == actual)
                
                # Update metrics
                self.signal_performance[signal_type]['total'] += 1
                if correct:
                    self.signal_performance[signal_type]['correct'] += 1
                    
        # Optionally adjust weights based on performance
        self._adjust_weights()
        
    def _adjust_weights(self):
        """Dynamically adjust signal weights based on performance"""
        # Only adjust if we have enough data
        min_samples = 20
        
        if all(self.signal_performance[k]['total'] >= min_samples for k in self.signal_weights):
            # Calculate accuracy for each signal type
            accuracies = {}
            for signal_type, metrics in self.signal_performance.items():
                if metrics['total'] > 0:
                    accuracies[signal_type] = metrics['correct'] / metrics['total']
                else:
                    accuracies[signal_type] = 0.33  # Default accuracy (random)
            
            # Normalize accuracies
            total_accuracy = sum(accuracies.values())
            if total_accuracy > 0:
                # New weights based on relative accuracy
                new_weights = {k: acc / total_accuracy for k, acc in accuracies.items()}
                
                # Smooth transition (weighted average with old weights)
                alpha = 0.2  # Learning rate
                for k in self.signal_weights:
                    if k in new_weights:
                        self.signal_weights[k] = (1 - alpha) * self.signal_weights[k] + alpha * new_weights[k]
                
                logger.info(f"Adjusted signal weights: {self.signal_weights}")