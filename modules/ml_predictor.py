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
import hashlib

logger = logging.getLogger(__name__)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and standardize column names"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    df.columns = [str(col).strip() for col in df.columns]
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced with forward-filling and better NaN handling"""
    df = df.copy()
    
    # Basic indicators
    for window in [5, 10, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # RSI with forward-fill
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean().ffill()
    avg_loss = loss.rolling(14).mean().ffill()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # MACD with forward-fill
    df['MACD'] = (df['Close'].ewm(span=12, adjust=False).mean() - 
                 df['Close'].ewm(span=26, adjust=False).mean()).ffill()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().ffill()
    
    # Bollinger Bands with forward-fill
    df['BB_Middle'] = df['Close'].rolling(20).mean().ffill()
    df['BB_Std'] = df['Close'].rolling(20).std().ffill()
    
    # Forward-fill all remaining NaNs
    return df.ffill().fillna(0)

def create_target_features(df: pd.DataFrame, days_ahead=1) -> pd.DataFrame:
    """Enhanced with length validation"""
    df = df.copy()
    df['target_return'] = df['Close'].pct_change(days_ahead).shift(-days_ahead)
    
    # Ensure we maintain consistent length
    if len(df) < days_ahead:
        raise ValueError("Insufficient data for target creation")
    
    return df.ffill().fillna(0)

def preprocess_data(df: pd.DataFrame, target='target_return', test_size=0.2):
    """Modified to maintain consistent lengths"""
    df = flatten_columns(df)
    df = add_technical_indicators(df)
    df = create_target_features(df)
    
    # Forward-fill instead of dropna
    df = df.ffill().fillna(0)
    
    X = df.drop(columns=[c for c in df.columns if c.startswith('target_')])
    y = df[target]
    
    # Chronological split with length validation
    split_idx = int(len(df) * (1 - test_size))
    if split_idx < 1:
        raise ValueError("Insufficient data for train/test split")
        
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

def ensemble_predict(X_train, y_train, X_test, time_limit=1200):
    """Enhanced with length validation and alignment"""
    try:
        # Convert inputs to DataFrames if needed
        X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        
        # Create training data with forward-fill
        train_data = X_train.ffill().fillna(0).copy()
        train_data['target'] = y_train.values
        
        # Train model
        predictor = TabularPredictor(label='target').fit(
            train_data=train_data,
            time_limit=time_limit,
            presets='medium_quality'
        )
        
        # Make predictions
        predictions = predictor.predict(X_test.ffill().fillna(0))
        
        # Ensure length matches
        if len(predictions) != len(X_test):
            logger.warning(f"Adjusting predictions length from {len(predictions)} to {len(X_test)}")
            predictions = predictions[:len(X_test)] if len(predictions) > len(X_test) else np.pad(
                predictions, 
                (0, len(X_test) - len(predictions)), 
                mode='constant'
            )
            
        return predictions.values if isinstance(predictions, pd.Series) else predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)[:200]}")
        return np.zeros(len(X_test))