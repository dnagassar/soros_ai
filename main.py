# main.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from modules.signal_aggregator import aggregate_signals
from modules.data_acquisition import fetch_price_data
from modules.ml_predictor import ensemble_predict_wrapper
from modules.sentiment_analysis import aggregate_sentiments
from modules.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

def add_technical_indicators(price_data):
    """
    Add technical indicators to price data
    
    Parameters:
      - price_data: DataFrame with OHLCV data
      
    Returns:
      - DataFrame with indicators added
    """
    df = price_data.copy()
    
    # Ensure DataFrame has required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column {col} in price data")
            df[col] = df['Close'] if 'Close' in df.columns else 0
    
    # Moving Averages
    for window in [5, 10, 20, 50, 100]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # Calculate MA ratio with zero check - FIX HERE
    if df['MA50'].iloc[-1] != 0:
        ma_ratio = df['MA20'].iloc[-1] / df['MA50'].iloc[-1]
    else:
        logger.warning("MA50 value is zero, using default ratio value")
        ma_ratio = 1.0  # Use a reasonable default when denominator is zero
    
    # Store the ratio in the DataFrame
    df['MA_ratio'] = ma_ratio
    
    # Add other indicators (RSI, MACD, etc.)
    # ... (Add your other indicator calculations here)
    
    return df

def get_ml_signal(symbol, price_data):
    """
    Get ML prediction signal for a symbol
    """
    try:
        # Prepare features and train/test data
        X_train = price_data.iloc[:-30].copy()
        y_train = X_train['Close'].pct_change().shift(-1).iloc[:-1]
        X_test = price_data.iloc[-30:].copy()
        
        # Use wrapper function to properly handle DataFrame hashing - FIX HERE
        predictions = ensemble_predict_wrapper(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test
        )
        
        # Calculate signal from predictions
        return predictions.mean()
        
    except Exception as e:
        logger.error(f"Error getting ML signal for {symbol}: {e}")
        return 0.0  # Return neutral signal on error

def main():
    """Main execution function"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for symbol in symbols:
        try:
            # Fetch price data
            price_data = fetch_price_data(symbol, '2023-01-01', '2023-12-31')
            
            # Add technical indicators
            price_data = add_technical_indicators(price_data)
            
            # Get ML signal
            ml_signal = get_ml_signal(symbol, price_data)
            logger.info(f"ML Predictions: Mean={ml_signal:.1f}")
            
            # Get sentiment
            sentiment_result = aggregate_sentiments(f"Latest news for {symbol}")
            logger.info(f"Combined Social Sentiment: {sentiment_result}")
            
            # Aggregate signals
            X_train = price_data.iloc[:-30].copy()
            y_train = X_train['Close'].pct_change().shift(-1).iloc[:-1]
            X_test = price_data.iloc[-30:].copy()
            
            final_signal = aggregate_signals(
                "News headline", 
                1, 
                symbol, 
                X_train, 
                y_train, 
                X_test, 
                [1, 1, 5, 10, 2]
            )
            
            logger.info(f"Final Aggregated Signal: {final_signal}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()