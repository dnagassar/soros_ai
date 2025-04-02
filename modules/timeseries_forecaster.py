# modules/timeseries_forecaster.py
import numpy as np
import pandas as pd
import torch
import logging
import os
from datetime import datetime
import pickle
import tempfile
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Import specialized libraries
try:
    from darts import TimeSeries
    from darts.models import TFTModel, NBEATSModel, TCNModel
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.likelihood_models import GaussianLikelihood
    from darts.metrics import mape
    
    DARTS_AVAILABLE = True
except ImportError:
    logger.warning("Darts library not available. Advanced time series models disabled.")
    DARTS_AVAILABLE = False

class TimeSeriesForecaster:
    """
    Advanced time series forecasting using state-of-the-art models like TFT,
    N-BEATS, and TCN as alternatives to TimesNet for financial forecasting.
    """
    
    def __init__(self, model_dir='models/timeseries', model_type='nbeats'):
        """
        Initialize the time series forecaster
        
        Parameters:
            model_dir: Directory to store trained models
            model_type: Type of model to use ('nbeats', 'tft', 'tcn')
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.covariates_scalers = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        if not DARTS_AVAILABLE:
            logger.error("Cannot initialize TimeSeriesForecaster: required libraries missing")
    
    def _create_model(self, input_chunk_length, forecast_horizon):
        """Create a model based on the selected type"""
        if not DARTS_AVAILABLE:
            return None
            
        if self.model_type == 'nbeats':
            # N-BEATS is often best for univariate financial time series
            return NBEATSModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=forecast_horizon,
                generic_architecture=True,
                num_stacks=10,
                num_blocks=3,
                num_layers=4,
                layer_widths=512,
                n_epochs=100,
                nr_epochs_val_period=1,
                batch_size=32,
                model_name=f"nbeats_{datetime.now().strftime('%Y%m%d')}",
                force_reset=True,
                save_checkpoints=True,
                pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
            )
        elif self.model_type == 'tft':
            # Temporal Fusion Transformer - excellent when you have multiple covariates
            return TFTModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=forecast_horizon,
                hidden_size=64,
                lstm_layers=2,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=32,
                n_epochs=100,
                likelihood=GaussianLikelihood(),
                pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
            )
        elif self.model_type == 'tcn':
            # Temporal Convolutional Network
            return TCNModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=forecast_horizon,
                kernel_size=3,
                num_filters=64,
                dilation_base=2,
                weight_norm=True,
                dropout=0.2,
                batch_size=32,
                n_epochs=100,
                pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
            )
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return None
    
    def prepare_data(self, df, target_col='Close', feature_cols=None):
        """
        Prepare data for time series forecasting
        
        Parameters:
            df: DataFrame with time series data (must have a datetime index)
            target_col: Target column to forecast
            feature_cols: List of feature columns to use as covariates
            
        Returns:
            tuple: (TimeSeries target, TimeSeries covariates)
        """
        if not DARTS_AVAILABLE:
            logger.error("Darts library not available")
            return None, None
            
        try:
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex, attempting to convert")
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            
            # Create target series
            target = TimeSeries.from_series(df[target_col])
            
            # Create covariates if feature columns provided
            covariates = None
            if feature_cols and all(col in df.columns for col in feature_cols):
                covariates = TimeSeries.from_dataframe(df[feature_cols])
            
            return target, covariates
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None
    
    def train_model(self, symbol, df, target_col='Close', feature_cols=None, 
                  input_chunk_length=30, forecast_horizon=5):
        """
        Train a time series model for the given symbol
        
        Parameters:
            symbol: Symbol for the model
            df: DataFrame with time series data
            target_col: Target column to forecast
            feature_cols: List of feature columns to use as covariates
            input_chunk_length: Number of past observations to use
            forecast_horizon: Number of future observations to predict
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not DARTS_AVAILABLE:
            logger.error("Cannot train model: required libraries missing")
            return False
            
        try:
            logger.info(f"Training time series model for {symbol}")
            
            # Prepare data
            target, covariates = self.prepare_data(df, target_col, feature_cols)
            
            if target is None:
                logger.error(f"Failed to prepare data for {symbol}")
                return False
            
            # Create and fit scaler for target
            scaler = Scaler()
            target_scaled = scaler.fit_transform(target)
            self.scalers[symbol] = scaler
            
            # Create and fit scaler for covariates if available
            if covariates is not None:
                covariates_scaler = Scaler()
                covariates_scaled = covariates_scaler.fit_transform(covariates)
                self.covariates_scalers[symbol] = covariates_scaler
            else:
                covariates_scaled = None
            
            # Create model
            model = self._create_model(input_chunk_length, forecast_horizon)
            
            if model is None:
                logger.error(f"Failed to create model for {symbol}")
                return False
            
            # Train model
            if covariates_scaled is not None:
                model.fit(
                    series=target_scaled,
                    past_covariates=covariates_scaled,
                    verbose=True
                )
            else:
                model.fit(
                    series=target_scaled,
                    verbose=True
                )
            
            # Save model
            self.models[symbol] = model
            
            # Save model to disk
            model_path = os.path.join(self.model_dir, f"{symbol}_timeseries_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'input_chunk_length': input_chunk_length,
                    'forecast_horizon': forecast_horizon,
                    'trained_at': datetime.now()
                }, f)
            
            # Save model in a temporary directory (darts models save themselves)
            with tempfile.TemporaryDirectory() as temp_dir:
                model.save(os.path.join(temp_dir, f"{symbol}_model"))
                
                # Copy files to the model directory
                import shutil
                for file in os.listdir(os.path.join(temp_dir, f"{symbol}_model")):
                    src = os.path.join(temp_dir, f"{symbol}_model", file)
                    dst = os.path.join(self.model_dir, f"{symbol}_model_{file}")
                    shutil.copy2(src, dst)
            
            # Save scalers
            with open(os.path.join(self.model_dir, f"{symbol}_scalers.pkl"), 'wb') as f:
                pickle.dump({
                    'target_scaler': self.scalers[symbol],
                    'covariates_scaler': self.covariates_scalers.get(symbol)
                }, f)
            
            logger.info(f"Successfully trained time series model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training time series model for {symbol}: {e}")
            return False
    
    def load_model(self, symbol):
        """
        Load a trained model for the given symbol
        
        Parameters:
            symbol: Symbol to load model for
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if not DARTS_AVAILABLE:
            logger.error("Cannot load model: required libraries missing")
            return False
            
        try:
            # Check if model info exists
            model_info_path = os.path.join(self.model_dir, f"{symbol}_timeseries_model.pkl")
            if not os.path.exists(model_info_path):
                logger.warning(f"No model info found for {symbol}")
                return False
            
            # Load model info
            with open(model_info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Set model type from saved info
            self.model_type = model_info['model_type']
            
            # Create model instance
            model = self._create_model(
                model_info['input_chunk_length'],
                model_info['forecast_horizon']
            )
            
            if model is None:
                logger.error(f"Failed to create model for {symbol}")
                return False
            
            # Load model from files
            try:
                model_files_prefix = os.path.join(self.model_dir, f"{symbol}_model_")
                # Find all files with this prefix
                model_files = [f for f in os.listdir(self.model_dir) if f.startswith(f"{symbol}_model_")]
                
                if not model_files:
                    logger.warning(f"No model files found for {symbol}")
                    return False
                
                # Create temporary directory with the expected structure
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_dir = os.path.join(temp_dir, f"{symbol}_model")
                    os.makedirs(model_dir)
                    
                    # Copy files to temp directory
                    import shutil
                    for file in model_files:
                        src = os.path.join(self.model_dir, file)
                        dst = os.path.join(model_dir, file.replace(f"{symbol}_model_", ""))
                        shutil.copy2(src, dst)
                    
                    # Load model
                    model = model.load(model_dir)
            except Exception as e:
                logger.error(f"Error loading model files for {symbol}: {e}")
                return False
            
            # Load scalers
            scalers_path = os.path.join(self.model_dir, f"{symbol}_scalers.pkl")
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scalers[symbol] = scalers['target_scaler']
                    if scalers['covariates_scaler'] is not None:
                        self.covariates_scalers[symbol] = scalers['covariates_scaler']
            
            # Store model
            self.models[symbol] = model
            
            logger.info(f"Successfully loaded time series model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading time series model for {symbol}: {e}")
            return False
    
    @lru_cache(maxsize=32)
    def predict(self, symbol, df, target_col='Close', feature_cols=None, n_forecast=5):
        """
        Make predictions with a trained model
        
        Parameters:
            symbol: Symbol to make predictions for
            df: DataFrame with recent data
            target_col: Target column to forecast
            feature_cols: List of feature columns to use as covariates
            n_forecast: Number of steps to forecast
            
        Returns:
            pd.Series: Forecast values
        """
        if not DARTS_AVAILABLE:
            logger.error("Cannot predict: required libraries missing")
            return None
            
        try:
            # Load model if not already loaded
            if symbol not in self.models:
                success = self.load_model(symbol)
                if not success:
                    logger.error(f"Failed to load model for {symbol}")
                    return None
            
            # Prepare data
            target, covariates = self.prepare_data(df, target_col, feature_cols)
            
            if target is None:
                logger.error(f"Failed to prepare data for {symbol}")
                return None
            
            # Scale data
            if symbol in self.scalers:
                target_scaled = self.scalers[symbol].transform(target)
                
                if covariates is not None and symbol in self.covariates_scalers:
                    covariates_scaled = self.covariates_scalers[symbol].transform(covariates)
                else:
                    covariates_scaled = None
            else:
                logger.error(f"No scaler found for {symbol}")
                return None
            
            # Make prediction
            model = self.models[symbol]
            if covariates_scaled is not None:
                forecast = model.predict(
                    n=n_forecast,
                    series=target_scaled,
                    past_covariates=covariates_scaled
                )
            else:
                forecast = model.predict(
                    n=n_forecast,
                    series=target_scaled
                )
            
            # Unscale forecast
            forecast = self.scalers[symbol].inverse_transform(forecast)
            
            # Convert to pandas series
            forecast_values = forecast.pd_series()
            
            logger.info(f"Made forecast for {symbol}: {forecast_values.values}")
            return forecast_values
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def calculate_signal(self, symbol, df, target_col='Close', feature_cols=None, n_forecast=5):
        """
        Calculate a trading signal based on the forecast
        
        Parameters:
            symbol: Symbol to calculate signal for
            df: DataFrame with recent data
            target_col: Target column to forecast
            feature_cols: List of feature columns to use as covariates
            n_forecast: Number of steps to forecast
            
        Returns:
            float: Signal between -1 and 1
        """
        if not DARTS_AVAILABLE:
            logger.error("Cannot calculate signal: required libraries missing")
            return 0.0
            
        try:
            # Get forecast
            forecast = self.predict(symbol, df, target_col, feature_cols, n_forecast)
            
            if forecast is None:
                logger.error(f"Failed to get forecast for {symbol}")
                return 0.0
            
            # Calculate signal
            # Use the trend of the forecast and its magnitude
            
            # Get current price
            current_price = df[target_col].iloc[-1]
            
            # Get forecasted prices
            forecast_prices = forecast.values
            
            # Calculate average forecasted return
            forecast_returns = [(price / current_price - 1) for price in forecast_prices]
            avg_forecast_return = np.mean(forecast_returns)
            
            # Calculate forecast trend
            if len(forecast_prices) > 1:
                trend = (forecast_prices[-1] / forecast_prices[0] - 1)
            else:
                trend = 0
            
            # Combine into signal (-1 to 1)
            # Scale by typical daily stock movement (e.g., 2% is significant)
            signal = (avg_forecast_return / 0.02) * 0.7 + (trend / 0.02) * 0.3
            
            # Clip to -1 to 1 range
            signal = max(min(signal, 1.0), -1.0)
            
            logger.info(f"Calculated time series signal for {symbol}: {signal:.4f}")
            return float(signal)
            
        except Exception as e:
            logger.error(f"Error calculating signal for {symbol}: {e}")
            return 0.0