# modules/signal_aggregator.py (improved version)
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import pickle
import os
import json
import hashlib

logger = logging.getLogger(__name__)

class SignalAggregator:
    """
    Improved Signal Aggregator with dynamic weighting and validation
    """
    
    def __init__(self, cache_dir='cache'):
        """Initialize the signal aggregator with improved caching"""
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initial signal weights
        self.signal_weights = {
            'sentiment': 0.20,
            'technical': 0.35,
            'ml': 0.25,
            'macro': 0.15,
            'earnings': 0.05
        }
        
        # Target weights - what we want to converge to based on performance
        self.target_weights = self.signal_weights.copy()
        
        # Signal performance tracking
        self.signal_performance = {
            'sentiment': {'correct': 0, 'total': 0},
            'technical': {'correct': 0, 'total': 0},
            'ml': {'correct': 0, 'total': 0},
            'macro': {'correct': 0, 'total': 0},
            'earnings': {'correct': 0, 'total': 0}
        }
        
        # Cached signals
        self.signal_cache = {}
        self.signal_age = {}
        
        # Cache TTL in seconds
        self.cache_ttl = {
            'sentiment': 43200,   # 12 hours
            'technical': 3600,    # 1 hour
            'ml': 86400,          # 1 day
            'macro': 604800,      # 1 week
            'earnings': 86400     # 1 day
        }
        
        # Minimum acceptable signal quality
        self.min_signal_quality = 0.4  # Minimum hit rate to consider a signal
        
        logger.info("Signal Aggregator initialized with improved weighting")
    
    def _cache_key(self, signal_type, symbol):
        """Generate a unique cache key with hash for better organization"""
        base_key = f"{signal_type}_{symbol}"
        return hashlib.md5(base_key.encode()).hexdigest()
    
    def load_cached_signal(self, signal_type, symbol):
        """Load a cached signal with improved error handling"""
        cache_key = self._cache_key(signal_type, symbol)
        
        # Check in-memory cache first (faster)
        if cache_key in self.signal_cache and cache_key in self.signal_age:
            age = (datetime.now() - self.signal_age[cache_key]).total_seconds()
            if age < self.cache_ttl.get(signal_type, 3600):
                return self.signal_cache[cache_key], age
        
        # Check disk cache
        if not self.cache_dir:
            return None, None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pickle")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                timestamp = cached_data.get('timestamp')
                signal = cached_data.get('signal')
                
                if timestamp and signal is not None:
                    age = (datetime.now() - timestamp).total_seconds()
                    if age < self.cache_ttl.get(signal_type, 3600):
                        # Update in-memory cache
                        self.signal_cache[cache_key] = signal
                        self.signal_age[cache_key] = timestamp
                        return signal, age
            except Exception as e:
                logger.warning(f"Error loading cached signal: {e}")
        
        return None, None
    
    def save_cached_signal(self, signal_type, symbol, signal):
        """Save a signal to cache with improved error handling"""
        cache_key = self._cache_key(signal_type, symbol)
        timestamp = datetime.now()
        
        # Update in-memory cache
        self.signal_cache[cache_key] = signal
        self.signal_age[cache_key] = timestamp
        
        # Save to disk cache
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pickle")
        try:
            cached_data = {
                'timestamp': timestamp,
                'signal': signal,
                'symbol': symbol,
                'type': signal_type
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logger.warning(f"Error saving cached signal: {e}")
    
    def get_sentiment_signal(self, symbol, news_text=None, social_query=None, force_refresh=False):
        """Get sentiment signal with improved validation"""
        # Implementation of sentiment signal generation
        # This would call the sentiment analysis module
        # For this example, I'll just return a placeholder signal
        return 0.3  # Placeholder positive sentiment
    
    def get_technical_signal(self, price_data, lookback=20):
        """Calculate technical indicator signals with improved normalization"""
        if price_data is None or price_data.empty or len(price_data) < lookback:
            logger.warning("Insufficient price data for technical signal")
            return 0.0
        
        try:
            # Indicator calculations would go here
            # For this example, I'll return a placeholder signal
            return 0.2  # Placeholder positive technical signal
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0
    
    def get_ml_signal(self, symbol, X_train, y_train, X_test, force_refresh=False):
        """Get ML prediction signal with improved validation"""
        # Implementation of ML signal generation
        # This would call the ML predictor module
        # For this example, I'll just return a placeholder signal
        return 0.1  # Placeholder positive ML signal
    
    def get_macro_signal(self, symbol=None, force_refresh=False):
        """Get macroeconomic signal with improved context"""
        # Implementation of macro signal generation
        # This would call the macro module
        # For this example, I'll just return a placeholder signal
        return 0.0  # Placeholder neutral macro signal
    
    def get_earnings_signal(self, symbol, force_refresh=False):
        """Get earnings-related signal with improved event detection"""
        # Implementation of earnings signal generation
        # This would call the earnings module
        # For this example, I'll just return a placeholder signal
        return 0.0  # Placeholder neutral earnings signal
    
    def update_signal_performance(self, signal_type, was_correct):
        """Update signal performance tracking"""
        if signal_type in self.signal_performance:
            self.signal_performance[signal_type]['total'] += 1
            if was_correct:
                self.signal_performance[signal_type]['correct'] += 1
                
            # Update target weights based on performance
            self._update_target_weights()
    
    def _update_target_weights(self):
        """Update target weights based on signal performance"""
        # Calculate hit rates for each signal type
        hit_rates = {}
        total_hit_rate = 0
        
        for signal_type, perf in self.signal_performance.items():
            if perf['total'] >= 10:  # Need minimum samples for reliability
                hit_rate = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                hit_rates[signal_type] = max(hit_rate, self.min_signal_quality)
                total_hit_rate += hit_rates[signal_type]
        
        # Set target weights based on relative hit rates
        if total_hit_rate > 0:
            for signal_type in hit_rates:
                self.target_weights[signal_type] = hit_rates[signal_type] / total_hit_rate
            
            # Gradually adjust current weights toward target (smoothing)
            for signal_type in self.signal_weights:
                if signal_type in self.target_weights:
                    # Move 10% toward target each time
                    self.signal_weights[signal_type] = (
                        0.9 * self.signal_weights[signal_type] + 
                        0.1 * self.target_weights[signal_type]
                    )
            
            # Normalize weights to sum to 1
            total_weight = sum(self.signal_weights.values())
            if total_weight > 0:
                for signal_type in self.signal_weights:
                    self.signal_weights[signal_type] /= total_weight
    
    def get_signal_quality(self):
        """Get quality metrics for each signal type"""
        quality = {}
        
        for signal_type, perf in self.signal_performance.items():
            if perf['total'] > 0:
                quality[signal_type] = {
                    'hit_rate': perf['correct'] / perf['total'],
                    'sample_size': perf['total'],
                    'weight': self.signal_weights.get(signal_type, 0)
                }
        
        return quality
    
    def aggregate_signals(self, symbol, price_data=None, news_text=None, social_query=None, 
                         X_train=None, y_train=None, X_test=None, signal_weights=None,
                         force_refresh=False):
        """
        Combine multiple signals with dynamic weighting and validation
        
        Returns:
          - dict: Aggregated signal and components with quality metrics
        """
        # Use custom weights if provided, otherwise use learned weights
        weights = signal_weights or self.signal_weights
        
        # Get individual signals
        sentiment_signal = self.get_sentiment_signal(symbol, news_text, social_query, force_refresh)
        
        technical_signal = 0.0
        if price_data is not None and not price_data.empty:
            technical_signal = self.get_technical_signal(price_data)
            
        ml_signal = 0.0
        if X_train is not None and y_train is not None and X_test is not None:
            ml_signal = self.get_ml_signal(symbol, X_train, y_train, X_test, force_refresh)
        
        macro_signal = self.get_macro_signal(symbol, force_refresh)
        earnings_signal = self.get_earnings_signal(symbol, force_refresh)
        
        # Create signal components dictionary
        signal_components = {
            'sentiment': sentiment_signal,
            'technical': technical_signal,
            'ml': ml_signal,
            'macro': macro_signal,
            'earnings': earnings_signal
        }
        
        # Calculate signal statistics
        signal_stats = {
            'mean': np.mean(list(signal_components.values())),
            'std': np.std(list(signal_components.values())),
            'min': min(signal_components.values()),
            'max': max(signal_components.values()),
            'agreement': self._calculate_agreement(signal_components)
        }
        
        # Calculate combined signal with current weights
        combined_signal = sum(signal * weights.get(name, 0.0) 
                             for name, signal in signal_components.items())
        
        # Ensure combined signal is in -1 to 1 range
        combined_signal = min(max(combined_signal, -1), 1)
        
        # Apply confidence adjustment based on agreement
        confidence = signal_stats['agreement']
        
        # Return result with components and metadata
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': combined_signal,
            'components': signal_components,
            'weights': weights,
            'stats': signal_stats,
            'confidence': confidence
        }
    
    def _calculate_agreement(self, signal_components):
        """
        Calculate agreement between different signals
        
        Returns:
          - float: Agreement score (0-1)
        """
        # Count how many signals have the same sign
        values = list(signal_components.values())
        if not values:
            return 0
            
        positive_count = sum(1 for v in values if v > 0.1)
        negative_count = sum(1 for v in values if v < -0.1)
        neutral_count = len(values) - positive_count - negative_count
        
        # If all signals agree, high confidence
        if positive_count == len(values) or negative_count == len(values):
            return 1.0
            
        # If most signals agree, medium confidence
        max_agreement = max(positive_count, negative_count, neutral_count)
        return max_agreement / len(values)