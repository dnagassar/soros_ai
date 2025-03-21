# modules/signal_aggregator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import pickle
import os
import json
from modules.sentiment_analysis import aggregate_sentiments
from modules.news_social_monitor import get_combined_sentiment
from modules.macro_module import get_gdp_indicator
from modules.earnings_module import get_upcoming_earnings
from modules.ml_predictor import ensemble_predict, add_technical_indicators, predict_with_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalAggregator:
    """
    Enhanced Signal Aggregator that combines multiple signal sources
    with adaptive weighting and caching.
    """
    
    def __init__(self, cache_dir='cache'):
        """
        Initialize the signal aggregator with optional caching
        
        Parameters:
          - cache_dir: Directory to store cached signals
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.signal_weights = {
            'sentiment': 0.25,
            'technical': 0.25,
            'ml': 0.25,
            'macro': 0.15,
            'earnings': 0.10
        }
        
        self.signal_cache = {}
        self.signal_age = {}
        self.cache_ttl = {
            'sentiment': 86400,  # 1 day in seconds
            'technical': 3600,   # 1 hour
            'ml': 86400,         # 1 day
            'macro': 604800,     # 1 week
            'earnings': 86400    # 1 day
        }
        
        logger.info("Signal Aggregator initialized")
    
    def load_cached_signal(self, signal_type, symbol):
        """
        Load a cached signal if available and not expired
        
        Parameters:
          - signal_type: Type of signal to load
          - symbol: Asset symbol
          
        Returns:
          - tuple: (signal, age_in_seconds) or (None, None) if no valid cache
        """
        cache_key = f"{signal_type}_{symbol}"
        
        # Check in-memory cache first
        if cache_key in self.signal_cache and cache_key in self.signal_age:
            age = (datetime.now() - self.signal_age[cache_key]).total_seconds()
            if age < self.cache_ttl.get(signal_type, 3600):
                return self.signal_cache[cache_key], age
        
        # Check disk cache
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
                logger.error(f"Error loading cached signal: {e}")
        
        return None, None
    
    def save_cached_signal(self, signal_type, symbol, signal):
        """
        Save a signal to cache
        
        Parameters:
          - signal_type: Type of signal to save
          - symbol: Asset symbol
          - signal: The signal value to cache
        """
        cache_key = f"{signal_type}_{symbol}"
        timestamp = datetime.now()
        
        # Update in-memory cache
        self.signal_cache[cache_key] = signal
        self.signal_age[cache_key] = timestamp
        
        # Save to disk cache
        if self.cache_dir:
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
                logger.error(f"Error saving cached signal: {e}")
    
    def get_sentiment_signal(self, symbol, news_text=None, social_query=None, force_refresh=False):
        """
        Get sentiment signal from news and social media
        
        Parameters:
          - symbol: Asset symbol
          - news_text: Optional specific news text to analyze
          - social_query: Optional social media query string
          - force_refresh: Force refresh the signal
          
        Returns:
          - float: Sentiment signal (-1 to 1)
        """
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_signal, age = self.load_cached_signal('sentiment', symbol)
            if cached_signal is not None:
                logger.debug(f"Using cached sentiment signal for {symbol} (age: {age:.0f}s)")
                return cached_signal
        
        try:
            # Generate default news text if not provided
            if news_text is None:
                news_text = f"Latest market news for {symbol}"
            
            # Get sentiment from news
            base_sentiment = aggregate_sentiments(news_text)
            
            # Get social media sentiment if query provided
            if social_query:
                social_sentiment = get_combined_sentiment(social_query, symbol=symbol)
                
                # Combine sentiments (weighted)
                combined_score = base_sentiment.get('score', 0) * 0.6 + social_sentiment.get('score', 0) * 0.4
            else:
                combined_score = base_sentiment.get('score', 0)
            
            # Normalize to -1 to 1 range
            signal = max(min(combined_score, 1), -1)
            
            # Cache the result
            self.save_cached_signal('sentiment', symbol, signal)
            
            logger.info(f"Sentiment signal for {symbol}: {signal:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error getting sentiment signal for {symbol}: {e}")
            return 0.0  # Neutral on error
    
    def get_technical_signal(self, price_data, lookback=20):
        """
        Calculate technical indicator signals from price data
        
        Parameters:
          - price_data: DataFrame with OHLCV data
          - lookback: Number of periods to look back
          
        Returns:
          - float: Technical signal (-1 to 1)
        """
        if price_data is None or price_data.empty or len(price_data) < lookback:
            logger.warning("Insufficient price data for technical signal")
            return 0.0
        
        try:
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in price_data.columns for col in required_cols):
                logger.warning(f"Missing required columns in price data: {[col for col in required_cols if col not in price_data.columns]}")
                return 0.0
            
            # Add technical indicators
            price_data = add_technical_indicators(price_data.copy())
            
            # Calculate signals from each indicator (all normalized to -1 to 1 range)
            signals = []
            
            # 1. Moving Average Trend signal
            if 'MA20' in price_data.columns and 'MA50' in price_data.columns:
                ma_ratio = price_data['MA20'].iloc[-1] / price_data['MA50'].iloc[-1]
                ma_signal = min(max((ma_ratio - 1) * 10, -1), 1)  # Scale by 10 for sensitivity
                signals.append(ma_signal)
            
            # 2. RSI signal
            if 'RSI' in price_data.columns:
                rsi = price_data['RSI'].iloc[-1]
                if rsi > 70:
                    rsi_signal = -((rsi - 70) / 30)  # Overbought - negative signal
                elif rsi < 30:
                    rsi_signal = (30 - rsi) / 30  # Oversold - positive signal
                else:
                    rsi_signal = (rsi - 50) / 20  # Neutral zone
                signals.append(rsi_signal)
            
            # 3. MACD signal
            if all(col in price_data.columns for col in ['MACD', 'MACD_Signal']):
                macd = price_data['MACD'].iloc[-1]
                signal = price_data['MACD_Signal'].iloc[-1]
                macd_signal = min(max((macd - signal) * 5, -1), 1)  # Scale for sensitivity
                signals.append(macd_signal)
            
            # 4. Bollinger Band signal
            if all(col in price_data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                last_close = price_data['Close'].iloc[-1]
                bb_upper = price_data['BB_Upper'].iloc[-1]
                bb_lower = price_data['BB_Lower'].iloc[-1]
                bb_middle = price_data['BB_Middle'].iloc[-1]
                
                # Normalize position within bands
                band_width = bb_upper - bb_lower
                if band_width > 0:
                    relative_pos = (last_close - bb_middle) / (band_width / 2)
                    bb_signal = -relative_pos  # Higher in band = more negative signal
                else:
                    bb_signal = 0
                signals.append(bb_signal)
            
            # 5. Price momentum signal
            returns = price_data['Close'].pct_change(5).iloc[-1] * 20  # 5-day return, scaled
            momentum_signal = min(max(returns, -1), 1)
            signals.append(momentum_signal)
            
            # 6. Volume signal
            if 'OBV' in price_data.columns:
                obv = price_data['OBV'].iloc[-1]
                obv_ma = price_data['OBV'].rolling(10).mean().iloc[-1]
                vol_signal = min(max((obv / obv_ma - 1) * 5, -1), 1)  # Scale for sensitivity
                signals.append(vol_signal * 0.5)  # Reduce volume impact
            
            # Calculate weighted average signal
            if signals:
                # Different weights for different types of signals
                weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05][:len(signals)]
                # Normalize weights
                weights = [w/sum(weights) for w in weights]
                
                tech_signal = sum(s * w for s, w in zip(signals, weights))
                logger.debug(f"Technical signals: {signals}, weights: {weights}, final: {tech_signal:.2f}")
                
                return tech_signal
            else:
                logger.warning("No technical signals calculated")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0
    
    def get_ml_signal(self, symbol, X_train, y_train, X_test, force_refresh=False):
        """
        Get machine learning prediction signal
        
        Parameters:
          - symbol: Asset symbol
          - X_train: Training features
          - y_train: Training target
          - X_test: Test features
          - force_refresh: Force refresh the signal
          
        Returns:
          - float: ML prediction signal (-1 to 1)
        """
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_signal, age = self.load_cached_signal('ml', symbol)
            if cached_signal is not None:
                logger.debug(f"Using cached ML signal for {symbol} (age: {age:.0f}s)")
                return cached_signal
        
        try:
            # Check if we have sufficient data
            if X_train is None or y_train is None or X_test is None:
                logger.warning(f"Insufficient data for ML prediction for {symbol}")
                return 0.0
            
            if len(X_train) < 30 or len(y_train) < 30:
                logger.warning(f"Training data too small for {symbol}: {len(X_train)} samples")
                return 0.0
            
            # Make predictions
            predictions = ensemble_predict(X_train, y_train, X_test, time_limit=600)
            
            if predictions is None or len(predictions) == 0:
                logger.warning(f"No predictions generated for {symbol}")
                return 0.0
            
            # For classification target (directional prediction)
            if np.all(np.isin(y_train.unique(), [-1, 0, 1])):
                # Average prediction (should be in range -1 to 1)
                avg_pred = predictions.mean()
                signal = min(max(avg_pred, -1), 1)
            
            # For regression target (return prediction)
            else:
                # Normalize prediction to -1 to 1 range
                # Assume predictions are daily returns, scale to make more sensitive
                pred_return = predictions.mean()
                signal = min(max(pred_return * 50, -1), 1)  # Scale by 50 for sensitivity
            
            # Cache the result
            self.save_cached_signal('ml', symbol, signal)
            
            logger.info(f"ML signal for {symbol}: {signal:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error getting ML signal for {symbol}: {e}")
            return 0.0
    
    def get_macro_signal(self, symbol=None, force_refresh=False):
        """
        Get macroeconomic signal
        
        Parameters:
          - symbol: Optional asset symbol (not used directly but for caching)
          - force_refresh: Force refresh the signal
          
        Returns:
          - float: Macro signal (-1 to 1)
        """
        # Use a common cache key for macro data regardless of symbol
        cache_symbol = 'MACRO' if symbol is None else symbol
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_signal, age = self.load_cached_signal('macro', cache_symbol)
            if cached_signal is not None:
                logger.debug(f"Using cached macro signal (age: {age:.0f}s)")
                return cached_signal
        
        try:
            # Get GDP indicator
            gdp_indicator = get_gdp_indicator()
            
            # Convert to signal
            if gdp_indicator == "BULLISH":
                signal = 0.75
            elif gdp_indicator == "BEARISH":
                signal = -0.75
            else:  # NEUTRAL
                signal = 0.0
            
            # Cache the result
            self.save_cached_signal('macro', cache_symbol, signal)
            
            logger.info(f"Macro signal: {signal:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error getting macro signal: {e}")
            return 0.0
    
    def get_earnings_signal(self, symbol, force_refresh=False):
        """
        Get earnings-related signal
        
        Parameters:
          - symbol: Asset symbol
          - force_refresh: Force refresh the signal
          
        Returns:
          - float: Earnings signal (-1 to 1)
        """
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_signal, age = self.load_cached_signal('earnings', symbol)
            if cached_signal is not None:
                logger.debug(f"Using cached earnings signal for {symbol} (age: {age:.0f}s)")
                return cached_signal
        
        try:
            # Get upcoming earnings
            earnings_status = get_upcoming_earnings(symbol)
            
            # Convert to signal
            if earnings_status == "EVENT_PENDING":
                # Approaching earnings - usually increases volatility
                signal = 0.5  # Slightly bullish expectation
            else:
                # No upcoming earnings
                signal = 0.0
            
            # Cache the result
            self.save_cached_signal('earnings', symbol, signal)
            
            logger.info(f"Earnings signal for {symbol}: {signal:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error getting earnings signal for {symbol}: {e}")
            return 0.0
    
    def aggregate_signals(self, symbol, price_data=None, news_text=None, social_query=None, 
                         X_train=None, y_train=None, X_test=None, signal_weights=None,
                         force_refresh=False):
        """
        Combine multiple signals into a final trading signal with adaptive weighting
        
        Parameters:
          - symbol: Asset symbol
          - price_data: DataFrame with OHLCV data (for technical signal)
          - news_text: News text for sentiment analysis
          - social_query: Query for social media sentiment
          - X_train, y_train, X_test: Data for ML prediction
          - signal_weights: Optional custom weights for signals
          - force_refresh: Force refresh all signals
          
        Returns:
          - dict: Aggregated signal and components
        """
        # Use custom weights if provided, otherwise use defaults
        weights = signal_weights or self.signal_weights
        
        # Get individual signals
        sentiment_signal = self.get_sentiment_signal(symbol, news_text, social_query, force_refresh)
        
        technical_signal = 0.0
        if price_data is not None and not price_data.empty:
            technical_signal = self.get_technical_signal(price_data)
            
            # Cache this signal
            self.save_cached_signal('technical', symbol, technical_signal)
        
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
        
        # Calculate combined signal
        combined_signal = sum(signal * weights.get(name, 0.0) 
                             for name, signal in signal_components.items())
        
        # Ensure combined signal is in -1 to 1 range
        combined_signal = min(max(combined_signal, -1), 1)
        
        # Log the result
        logger.info(f"Aggregated signal for {symbol}: {combined_signal:.2f}")
        logger.debug(f"Signal components: {json.dumps({k: f'{v:.2f}' for k, v in signal_components.items()})}")
        
        # Return result with components
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': combined_signal,
            'components': signal_components,
            'weights': weights
        }
    
    def adjust_weights(self, performance_history=None):
        """
        Dynamically adjust signal weights based on historical performance
        
        Parameters:
          - performance_history: Optional dict with signal-wise performance metrics
          
        Returns:
          - dict: Updated weights
        """
        if performance_history is None:
            logger.debug("No performance history provided, using default weights")
            return self.signal_weights
        
        try:
            # Example implementation: adjust weights based on signal accuracy
            new_weights = {}
            total_accuracy = sum(performance_history.values())
            
            if total_accuracy > 0:
                for signal_type, accuracy in performance_history.items():
                    if signal_type in self.signal_weights:
                        # Adjusted weight based on relative accuracy
                        new_weights[signal_type] = accuracy / total_accuracy
            else:
                new_weights = self.signal_weights
            
            logger.info(f"Adjusted weights: {json.dumps({k: f'{v:.2f}' for k, v in new_weights.items()})}")
            
            # Update internal weights
            self.signal_weights = new_weights
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Error adjusting weights: {e}")
            return self.signal_weights
    
    def clear_cache(self, symbol=None, signal_type=None):
        """
        Clear cached signals
        
        Parameters:
          - symbol: Optional symbol to clear (None for all)
          - signal_type: Optional signal type to clear (None for all)
        """
        # Clear in-memory cache
        keys_to_remove = []
        
        for key in list(self.signal_cache.keys()):
            parts = key.split('_', 1)
            if len(parts) == 2:
                cache_type, cache_symbol = parts
                
                if (signal_type is None or cache_type == signal_type) and \
                   (symbol is None or cache_symbol == symbol):
                    keys_to_remove.append(key)
        
        # Remove from in-memory cache
        for key in keys_to_remove:
            if key in self.signal_cache:
                del self.signal_cache[key]
            if key in self.signal_age:
                del self.signal_age[key]
        
        # Clear disk cache
        if self.cache_dir and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pickle'):
                    parts = filename[:-7].split('_', 1)
                    if len(parts) == 2:
                        cache_type, cache_symbol = parts
                        
                        if (signal_type is None or cache_type == signal_type) and \
                           (symbol is None or cache_symbol == symbol):
                            try:
                                os.remove(os.path.join(self.cache_dir, filename))
                            except Exception as e:
                                logger.error(f"Error removing cache file {filename}: {e}")
        
        logger.info(f"Cleared {len(keys_to_remove)} cached signals")

def aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test, signal_ages,
                     social_query=None):
    """
    Legacy function to maintain backward compatibility with original code
    """
    # Create a SignalAggregator instance
    aggregator = SignalAggregator()
    
    # Load price data for the symbol
    price_data = None
    if isinstance(X_test, pd.DataFrame) and not X_test.empty:
        # Try to use X_test as price data
        price_data = X_test.copy()
    
    # Convert the technical_signal to a numerical value if it's not already
    if not isinstance(technical_signal, (int, float)):
        try:
            technical_signal = float(technical_signal)
        except (ValueError, TypeError):
            technical_signal = 0
    
    # Override the technical signal if price data is available
    if technical_signal != 0 and price_data is None:
        # Create a temporary price data with just enough to override the technical signal
        dummy_data = {'Close': [100] * 50}
        for i in range(1, 50):
            dummy_data['Close'][i] = dummy_data['Close'][i-1] * (1 + np.random.normal(0, 0.005))
        
        price_data = pd.DataFrame(dummy_data)
        aggregator.signal_cache['technical_' + symbol] = technical_signal
    
    # Prepare signal ages if provided
    if signal_ages and len(signal_ages) >= 5:
        # Expected order: sentiment, technical, macro, earnings, ml
        current_time = datetime.now()
        
        for i, (signal_type, age) in enumerate([
            ('sentiment', signal_ages[0]), 
            ('technical', signal_ages[1]), 
            ('macro', signal_ages[2]), 
            ('earnings', signal_ages[3]), 
            ('ml', signal_ages[4])
        ]):
            if age > 0:
                cache_key = f"{signal_type}_{symbol}"
                aggregator.signal_age[cache_key] = current_time - timedelta(days=age)
    
    # Aggregate the signals
    result = aggregator.aggregate_signals(
        symbol=symbol,
        price_data=price_data,
        news_text=news_text,
        social_query=social_query,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test
    )
    
    # Return just the final signal for backward compatibility
    return result['signal']

if __name__ == "__main__":
    # Example usage
    aggregator = SignalAggregator(cache_dir='cache')
    
    # Test with mock data
    symbol = 'AAPL'
    news_text = "Apple reports record earnings."
    social_query = "AAPL stock"
    
    # Create dummy price data
    dates = pd.date_range(start='2022-01-01', periods=100)
    price_data = pd.DataFrame({
        'Open': np.random.normal(150, 5, 100),
        'High': np.random.normal(155, 5, 100),
        'Low': np.random.normal(145, 5, 100),
        'Close': np.random.normal(150, 5, 100),
        'Volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Create dummy ML data
    X_train = price_data.iloc[:-30].copy()
    y_train = X_train['Close'].pct_change().shift(-1).iloc[:-1]
    X_test = price_data.iloc[-30:].copy()
    
    # Get aggregated signal
    result = aggregator.aggregate_signals(
        symbol=symbol,
        price_data=price_data,
        news_text=news_text,
        social_query=social_query,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        force_refresh=True
    )
    
    print(f"Aggregated signal for {symbol}: {result['signal']:.2f}")
    print("Signal components:")
    for name, value in result['components'].items():
        print(f"  {name}: {value:.2f}")
    
    # Test legacy function
    signal_ages = [1, 1, 5, 10, 2]
    legacy_signal = aggregate_signals(
        news_text=news_text,
        technical_signal=0.5,
        symbol=symbol,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        signal_ages=signal_ages,
        social_query=social_query
    )
    
    print(f"Legacy function signal: {legacy_signal:.2f}")