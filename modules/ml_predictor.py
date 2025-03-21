# modules/ml_predictor.py
import pandas as pd
import numpy as np
import logging
from autogluon.tabular import TabularPredictor
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
from datetime import datetime
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the DataFrame has MultiIndex columns (e.g. from yfinance),
    flatten them to a single level by choosing the first non-empty level.
    Also convert all column names to strings and strip whitespace.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    df.columns = [str(col).strip() for col in df.columns]
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame to improve prediction quality.
    
    Parameters:
      - df (pd.DataFrame): DataFrame with OHLCV data
      
    Returns:
      - pd.DataFrame: DataFrame with additional technical indicators
    """
    # Ensure we're working with a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.warning(f"Missing columns for technical indicators: {missing}")
        return df
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # Exponential Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    pos_flow_sum = positive_flow.rolling(window=14).sum()
    neg_flow_sum = negative_flow.rolling(window=14).sum()
    
    money_ratio = pos_flow_sum / neg_flow_sum
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility (Standard Deviation of Returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def create_target_features(df: pd.DataFrame, days_ahead=1, target_column='Close') -> pd.DataFrame:
    """
    Create target variable for prediction and add features based on the target.
    
    Parameters:
      - df (pd.DataFrame): DataFrame with price data
      - days_ahead (int): Number of days ahead to predict
      - target_column (str): Column to use for prediction target
      
    Returns:
      - pd.DataFrame: DataFrame with target variable added
    """
    # Ensure we're working with a copy
    df = df.copy()
    
    # Add the raw target (future price)
    df['target_price'] = df[target_column].shift(-days_ahead)
    
    # Add the percentage change as target
    df['target_return'] = df['target_price'] / df[target_column] - 1
    
    # Add a classification target (up/down/neutral)
    threshold = 0.005  # 0.5% change threshold
    conditions = [
        (df['target_return'] > threshold),
        (df['target_return'] < -threshold),
        (True)  # For all other values (neutral)
    ]
    choices = [1, -1, 0]  # Up, Down, Neutral
    df['target_direction'] = np.select(conditions, choices, default=0)
    
    # Remove rows with NaN targets
    df = df.dropna(subset=['target_price', 'target_return'])
    
    return df

def preprocess_data(df: pd.DataFrame, target='target_return', scale=True, test_size=0.2):
    """
    Preprocess data for ML model training: flatten columns, add technical indicators,
    create targets, split into train/validation, and scale features.
    
    Parameters:
      - df (pd.DataFrame): Raw DataFrame with OHLCV data
      - target (str): Target column for prediction
      - scale (bool): Whether to scale the features
      - test_size (float): Proportion of data to use for testing
      
    Returns:
      - tuple: (X_train, X_val, y_train, y_val, scaler)
    """
    # Flatten columns if needed
    df = flatten_columns(df)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create target variables if not already present
    if not any(col.startswith('target_') for col in df.columns):
        df = create_target_features(df)
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Split features and target
    if target in df.columns:
        X = df.drop(columns=[col for col in df.columns if col.startswith('target_')])
        y = df[target]
    else:
        raise ValueError(f"Target column {target} not found in DataFrame")
    
    # Split into train and validation sets chronologically
    train_size = int(len(df) * (1 - test_size))
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
    
    # Scale features if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        columns_to_scale = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_val.loc[:, columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    
    return X_train, X_val, y_train, y_val, scaler

def check_and_prep_data(data: pd.DataFrame, label_column='target_return'):
    """
    Check if data has the required label column, and prepare it for AutoGluon.
    
    Parameters:
      - data (pd.DataFrame): The data to prepare
      - label_column (str): The target column name
      
    Returns:
      - pd.DataFrame: Prepared data with label column
    """
    data = flatten_columns(data.copy())
    
    # If the target column doesn't exist, try to create it
    if label_column not in data.columns:
        logger.warning(f"Label column {label_column} not found, attempting to create targets...")
        
        if 'Close' in data.columns:
            data = create_target_features(data)
        else:
            raise ValueError(f"Cannot create target: 'Close' column not found in data")
    
    # Check again after attempted creation
    if label_column not in data.columns:
        raise ValueError(f"Failed to create label column: {label_column}")
    
    # Remove rows with NaN in the label column
    data = data.dropna(subset=[label_column])
    
    # Make sure there are enough samples
    if len(data) < 10:
        raise ValueError(f"Insufficient data after preprocessing: {len(data)} rows")
    
    logger.info(f"Data prepared with shape: {data.shape}")
    return data

def train_ensemble_model(train_data: pd.DataFrame, label_column='target_return', 
                        time_limit=1200, hyperparameters=None):
    """
    Trains an AutoGluon ensemble model on the provided training data.
    Ensures that the DataFrame is flattened and that the label column exists.
    
    Parameters:
      - train_data (pd.DataFrame): Training data with features and the target
      - label_column (str): Name of the target variable
      - time_limit (int): Training time limit in seconds
      - hyperparameters (dict): Optional hyperparameters
      
    Returns:
      - predictor: The trained TabularPredictor model
    """
    if hyperparameters is None:
        hyperparameters = {}
    
    mlflow.start_run()
    
    try:
        # Prepare the data, creating targets if needed
        train_data = check_and_prep_data(train_data, label_column)
        
        # Log information about the training data
        logger.info(f"Training with data shape: {train_data.shape}")
        logger.info(f"Training columns: {train_data.columns.tolist()}")
        
        # Create a unique save path for the predictor
        model_path = os.path.join('models', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(model_path, exist_ok=True)
        
        # Set up predictor with robust configs
        predictor = TabularPredictor(
            label=label_column,
            eval_metric='rmse' if label_column == 'target_return' else 'accuracy',
            path=model_path
        )
        
        # Use a validation split to avoid training issues
        train_size = int(0.8 * len(train_data))
        train_subset, val_subset = train_data.iloc[:train_size], train_data.iloc[train_size:]
        
        # Fit the model with validation data to avoid RMSE=inf issues
        predictor_fit_kwargs = {
            'presets': 'medium_quality',
            'time_limit': time_limit,
            'verbosity': 2,
            'tuning_data': val_subset if len(val_subset) > 10 else None,
            'num_bag_folds': 5,
            'num_bag_sets': 1
        }
        
        if hyperparameters:
            predictor_fit_kwargs['hyperparameters'] = hyperparameters
        
        predictor.fit(train_data=train_subset, **predictor_fit_kwargs)
        
        # Validate the model
        validation_score = predictor.evaluate(val_subset)
        logger.info(f"Validation score: {validation_score}")
        
        # Log parameters and metrics to MLflow
        for model_name, model_info in predictor.info().get('model_info', {}).items():
            for param_name, param_value in model_info.get('hyperparameters', {}).items():
                # Convert complex types to strings for MLflow
                if not isinstance(param_value, (int, float, str, bool)):
                    param_value = str(param_value)
                mlflow.log_param(f"{model_name}_{param_name}", param_value)
        
        mlflow.log_metric("validation_score", validation_score)
        mlflow.log_metric("train_data_size", len(train_subset))
        mlflow.log_metric("validation_data_size", len(val_subset))
        
        # Save details about the dataset columns
        with open(os.path.join(model_path, 'columns.pkl'), 'wb') as f:
            pickle.dump(train_data.columns.tolist(), f)
        
        return predictor
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        mlflow.log_metric("training_error", 1)
        raise
    
    finally:
        mlflow.end_run()

def predict_with_model(predictor, test_data: pd.DataFrame):
    """
    Uses the trained predictor to make predictions on test_data.
    If no models were trained, returns a dummy Series of zeros.
    
    Parameters:
      - predictor: The trained TabularPredictor model
      - test_data (pd.DataFrame): Data for generating predictions
      
    Returns:
      - predictions: A pandas Series of predictions
    """
    # Ensure columns are flattened
    test_data = flatten_columns(test_data.copy())
    
    # Check if predictor has models
    if not predictor.info().get('model_info'):
        logger.warning("No models available; returning zeros as dummy predictions.")
        return pd.Series(np.zeros(len(test_data)))
    
    # Get the list of expected columns
    model_path = predictor.path
    expected_columns_path = os.path.join(model_path, 'columns.pkl')
    
    if os.path.exists(expected_columns_path):
        with open(expected_columns_path, 'rb') as f:
            expected_columns = pickle.load(f)
        
        # Ensure test data has the same columns
        missing_features = [col for col in expected_columns if col not in test_data.columns 
                           and not col.startswith('target_')]
        
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
            # Add missing features with zeros
            for col in missing_features:
                test_data[col] = 0
    
    try:
        # Make predictions
        predictions = predictor.predict(test_data)
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return pd.Series(np.zeros(len(test_data)))

@lru_cache(maxsize=8)
def ensemble_predict(X_train_key, y_train_key, X_test_key, time_limit=1200, hyperparameters=None):
    """
    Cached version of model training and prediction to avoid retraining.
    Note: this is a simplified version that uses tuple representation as keys due to lru_cache limitations.
    
    For real use, consider implementing a more sophisticated caching mechanism.
    """
    # Convert back to the original dataframes
    X_train = pd.DataFrame(np.array(X_train_key), columns=[f'feature_{i}' for i in range(len(X_train_key[0]))])
    y_train = pd.Series(y_train_key)
    X_test = pd.DataFrame(np.array(X_test_key), columns=[f'feature_{i}' for i in range(len(X_test_key[0]))])
    
    # Add the target column to the training data
    train_data = X_train.copy()
    train_data['target_return'] = y_train
    
    # Train the model
    predictor = train_ensemble_model(train_data, time_limit=time_limit, hyperparameters=hyperparameters)
    
    # Make predictions
    predictions = predict_with_model(predictor, X_test)
    
    return predictions.values

def ensemble_predict_wrapper(X_train, y_train, X_test, time_limit=1200, hyperparameters=None):
    """
    Wrapper for ensemble_predict that handles the conversion to hashable types for caching.
    
    Parameters:
      - X_train: Training features DataFrame
      - y_train: Training target Series
      - X_test: Test features DataFrame
      - time_limit: Time limit for training in seconds
      - hyperparameters: Optional hyperparameters dictionary
      
    Returns:
      - predictions: Predictions as a pandas Series
    """
    # Ensure inputs are DataFrames/Series
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Convert to tuples for caching
    X_train_key = tuple(map(tuple, X_train.values))
    y_train_key = tuple(y_train.values)
    X_test_key = tuple(map(tuple, X_test.values))
    
    # Call the cached function
    predictions = ensemble_predict(X_train_key, y_train_key, X_test_key, time_limit, 
                                  hyperparameters=tuple(hyperparameters.items()) if hyperparameters else None)
    
    return pd.Series(predictions, index=X_test.index)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage for debugging:
    logger.info("Loading test data...")
    
    try:
        df = pd.read_csv('../data/historical_prices.csv')
        df = flatten_columns(df)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Create target variables
        df = create_target_features(df)
        logger.info(f"Processed data with shape: {df.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val, scaler = preprocess_data(df)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Train model
        train_data = X_train.copy()
        train_data['target_return'] = y_train
        
        predictor = train_ensemble_model(train_data=train_data, time_limit=600)
        logger.info("Model training complete")
        
        # Make predictions
        preds = predict_with_model(predictor, X_val)
        
        # Evaluate
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        
        logger.info(f"Validation RMSE: {rmse:.4f}")
        logger.info(f"Validation MAE: {mae:.4f}")
        
    except Exception as e:
        logger.error(f"Error in testing: {e}")