# Create modules/ml_pipeline.py
import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
from datetime import datetime
import mlflow
import traceback

logger = logging.getLogger(__name__)

class MLPipeline:
    """
    Enhanced ML pipeline with feature engineering, 
    model training, and prediction capabilities
    """
    
    def __init__(self, model_dir='models'):
        """Initialize ML pipeline"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.feature_columns = {}
        
    def prepare_features(self, df, symbol, target_column='target_return', lookback_days=5):
        """
        Prepare features from price data
        
        Parameters:
          - df: DataFrame with price data
          - symbol: Asset symbol
          - target_column: Target column name
          - lookback_days: Days to look back for lagged features
          
        Returns:
          - tuple: (X, y, feature_names)
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, cannot prepare features")
            return None, None, None
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Ensure we have OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"Missing required column {col} for {symbol}")
                    return None, None, None
            
            # Create target variable
            if target_column not in data.columns:
                # Future return
                data[target_column] = data['Close'].pct_change(1).shift(-1)
            
            # Basic features
            # Price ratios
            data['high_low_ratio'] = data['High'] / data['Low']
            data['close_open_ratio'] = data['Close'] / data['Open']
            
            # Volume features
            data['volume_ma10'] = data['Volume'].rolling(10).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_ma10']
            
            # Returns
            data['return_1d'] = data['Close'].pct_change(1)
            data['return_5d'] = data['Close'].pct_change(5)
            data['return_10d'] = data['Close'].pct_change(10)
            data['return_20d'] = data['Close'].pct_change(20)
            
            # Volatility
            data['volatility_10d'] = data['return_1d'].rolling(10).std()
            data['volatility_20d'] = data['return_1d'].rolling(20).std()
            
            # Moving averages
            for ma_period in [5, 10, 20, 50, 100]:
                data[f'ma_{ma_period}'] = data['Close'].rolling(ma_period).mean()
                data[f'ma_ratio_{ma_period}'] = data['Close'] / data[f'ma_{ma_period}']
            
            # MACD
            data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['Close'].rolling(20).mean()
            data['bb_std'] = data['Close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
            data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Lagged features
            for lag in range(1, lookback_days + 1):
                data[f'close_lag_{lag}'] = data['Close'].shift(lag)
                data[f'return_lag_{lag}'] = data['return_1d'].shift(lag)
                data[f'high_low_ratio_lag_{lag}'] = data['high_low_ratio'].shift(lag)
                data[f'volume_ratio_lag_{lag}'] = data['volume_ratio'].shift(lag)
            
            # Date-based features
            if isinstance(data.index, pd.DatetimeIndex):
                data['day_of_week'] = data.index.dayofweek
                data['day_of_month'] = data.index.day
                data['month'] = data.index.month
            
            # Remove NaN values
            data.dropna(inplace=True)
            
            # Split features and target
            y = data[target_column]
            
            # Drop columns that shouldn't be features
            cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', target_column]
            feature_cols = [col for col in data.columns if col not in cols_to_drop]
            
            # Store feature columns for this symbol
            self.feature_columns[symbol] = feature_cols
            
            X = data[feature_cols]
            
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    def train_model(self, X, y, symbol, model_params=None, time_limit=600):
        """
        Train model using available ML framework
        
        Parameters:
          - X: Feature DataFrame
          - y: Target Series
          - symbol: Asset symbol
          - model_params: Optional model parameters
          - time_limit: Training time limit in seconds
          
        Returns:
          - tuple: (model, performance)
        """
        if X is None or y is None or X.empty or len(y) == 0:
            logger.warning(f"Insufficient data for model training for {symbol}")
            return None, None
        
        try:
            # Create directory for this symbol's models
            symbol_dir = os.path.join(self.model_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Start MLflow run
            run_name = f"{symbol}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log dataset info
            mlflow.log_param("data_rows", len(X))
            mlflow.log_param("feature_count", len(X.columns))
            
            # Try using AutoGluon if available
            try:
                from autogluon.tabular import TabularPredictor
                
                # Combine features and target for AutoGluon
                train_data = X.copy()
                train_data['target'] = y
                
                # Create model with AutoGluon
                model_path = os.path.join(symbol_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Default model parameters
                default_params = {
                    "presets": "medium_quality",
                    "time_limit": time_limit
                }
                
                # Merge with custom parameters if provided
                if model_params:
                    default_params.update(model_params)
                
                # Log parameters
                for param, value in default_params.items():
                    mlflow.log_param(param, value)
                
                # Initialize predictor
                predictor = TabularPredictor(
                    label='target',
                    path=model_path
                )
                
                # Train the model
                logger.info(f"Training model for {symbol} with {len(train_data)} samples using AutoGluon...")
                predictor.fit(
                    train_data=train_data,
                    **default_params
                )
                
                # Evaluate performance
                performance = predictor.evaluate(train_data)
                
                # Log metrics
                for metric, value in performance.items():
                    mlflow.log_metric(metric, value)
                
                # Save feature columns
                with open(os.path.join(model_path, 'feature_columns.json'), 'w') as f:
                    json.dump(self.feature_columns[symbol], f)
                
                # Store model reference
                self.models[symbol] = {
                    'predictor': predictor,
                    'path': model_path,
                    'performance': performance,
                    'trained_at': datetime.now().isoformat()
                }
                
                logger.info(f"Model training completed for {symbol}")
                
                return predictor, performance
                
            except ImportError:
                # Fall back to scikit-learn
                logger.info("AutoGluon not available, falling back to scikit-learn")
                
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error, r2_score
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Create model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Train model
                logger.info(f"Training model for {symbol} with {len(X_train)} samples using sklearn...")
                model.fit(X_train, y_train)
                
                # Evaluate performance
                y_pred = model.predict(X_val)
                performance = {
                    'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
                    'r2': float(r2_score(y_val, y_pred))
                }
                
                # Log metrics
                for metric, value in performance.items():
                    mlflow.log_metric(metric, value)
                
                # Save model
                model_path = os.path.join(symbol_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save feature columns
                feature_path = os.path.join(symbol_dir, f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(feature_path, 'w') as f:
                    json.dump(self.feature_columns[symbol], f)
                
                # Store model reference
                self.models[symbol] = {
                    'model': model,
                    'path': model_path,
                    'feature_path': feature_path,
                    'performance': performance,
                    'trained_at': datetime.now().isoformat()
                }
                
                logger.info(f"Model training completed for {symbol}")
                
                return model, performance
                
        except Exception as e:
            logger.error(f"Error during model training for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None, None
        
        finally:
            # End MLflow run
            try:
                mlflow.end_run()
            except:
                pass
    
    def predict(self, X, symbol):
        """
        Make predictions using trained model
        
        Parameters:
          - X: Feature DataFrame
          - symbol: Asset symbol
          
        Returns:
          - Series: Predictions
        """
        if symbol not in self.models:
            logger.warning(f"No trained model available for {symbol}")
            return None
        
        try:
            model_info = self.models[symbol]
            
            # Check which type of model we have
            if 'predictor' in model_info:
                # AutoGluon model
                predictor = model_info['predictor']
                predictions = predictor.predict(X)
            elif 'model' in model_info:
                # Scikit-learn model
                model = model_info['model']
                predictions = pd.Series(model.predict(X), index=X.index)
            else:
                logger.warning(f"Unknown model type for {symbol}")
                return None
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def save_model(self, symbol):
        """Save model metadata"""
        if symbol not in self.models:
            logger.warning(f"No model to save for {symbol}")
            return False
        
        try:
            # Model is already saved during training, just save metadata
            model_info = self.models[symbol].copy()
            
            # Remove non-serializable objects
            if 'predictor' in model_info:
                del model_info['predictor']
            if 'model' in model_info:
                del model_info['model']
            
            # Save metadata
            metadata_path = os.path.join(self.model_dir, f"{symbol}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=4, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            return False
    
    def load_model(self, symbol):
        """Load model for a symbol"""
        try:
            # Check for model directory
            symbol_dir = os.path.join(self.model_dir, symbol)
            if not os.path.exists(symbol_dir):
                logger.warning(f"No model directory found for {symbol}")
                return False
            
            # Find most recent model file (could be AutoGluon directory or sklearn pickle)
            items = [d for d in os.listdir(symbol_dir) 
                    if os.path.isdir(os.path.join(symbol_dir, d)) or d.endswith('.pkl')]
            
            if not items:
                logger.warning(f"No model found for {symbol}")
                return False
            
            # Sort by creation time (newest first)
            items.sort(key=lambda d: os.path.getmtime(os.path.join(symbol_dir, d)), reverse=True)
            latest_item = items[0]
            latest_path = os.path.join(symbol_dir, latest_item)
            
            # Try to load based on file type
            if latest_item.endswith('.pkl'):
                # Scikit-learn model
                with open(latest_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Find corresponding feature file
                feature_files = [f for f in os.listdir(symbol_dir) if f.startswith('features_') and f.endswith('.json')]
                if feature_files:
                    feature_files.sort(key=lambda f: os.path.getmtime(os.path.join(symbol_dir, f)), reverse=True)
                    feature_path = os.path.join(symbol_dir, feature_files[0])
                    
                    # Load feature columns
                    with open(feature_path, 'r') as f:
                        self.feature_columns[symbol] = json.load(f)
                
                # Store model reference
                self.models[symbol] = {
                    'model': model,
                    'path': latest_path,
                    'loaded_at': datetime.now().isoformat()
                }
                
                logger.info(f"Loaded scikit-learn model for {symbol} from {latest_path}")
                
            else:
                # AutoGluon model
                try:
                    from autogluon.tabular import TabularPredictor
                    predictor = TabularPredictor.load(latest_path)
                    
                    # Load feature columns
                    feature_path = os.path.join(latest_path, 'feature_columns.json')
                    if os.path.exists(feature_path):
                        with open(feature_path, 'r') as f:
                            self.feature_columns[symbol] = json.load(f)
                    
                    # Store model reference
                    self.models[symbol] = {
                        'predictor': predictor,
                        'path': latest_path,
                        'loaded_at': datetime.now().isoformat()
                    }
                    
                    logger.info(f"Loaded AutoGluon model for {symbol} from {latest_path}")
                except ImportError:
                    logger.error(f"AutoGluon not available, cannot load model from {latest_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return False