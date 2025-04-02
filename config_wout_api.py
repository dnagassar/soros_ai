# config.py with out sensitive API keys
import os
from enum import Enum

# API Keys (only fetched from environment variables)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Alpaca API settings for live trading
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'True').lower() == 'true'

# Dashboard settings
DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD')  # Don't set a default password!

class SystemConfig:
    """System configuration settings"""
    # Directories
    DEFAULT_DATA_DIR = os.getenv('DATA_DIR', 'data')
    DEFAULT_MODELS_DIR = os.getenv('MODELS_DIR', 'models')
    DEFAULT_RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
    DEFAULT_REPORTS_DIR = os.getenv('REPORTS_DIR', 'reports')
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    
    # Trading parameters
    MAX_ASSETS = int(os.getenv('MAX_ASSETS', '10'))
    
    # Dashboard settings
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8501'))

class GlobalSettings:
    """Global settings that can be modified at runtime"""
    # Trading mode: 'backtest', 'paper', 'live'
    TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
    
    # Risk tolerance: 'low', 'medium', 'high'
    RISK_TOLERANCE = os.getenv('RISK_TOLERANCE', 'medium')
    
    # Active strategy
    ACTIVE_STRATEGY = os.getenv('ACTIVE_STRATEGY', 'AdaptiveSentimentStrategy')
    
    # Paper trading balance
    PAPER_TRADING_BALANCE = float(os.getenv('PAPER_TRADING_BALANCE', '100000'))

def get_strategy_parameters(strategy_name):
    """Get parameters for a specific strategy"""
    if strategy_name == 'AdaptiveSentimentStrategy':
        return {
            'sentiment_period': 3,
            'vol_window': 20,
            'ema_short': 10,
            'ema_medium': 30,
            'ema_long': 50,
            'rsi_period': 14,
            'stop_loss': 0.03,
            'take_profit': 0.05,
            'risk_factor': 0.01,
            'ml_weight': 0.4,
            'sentiment_weight': 0.3,
            'tech_weight': 0.3
        }
    elif strategy_name == 'MACDStrategy':
        return {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stop_loss': 0.03
        }
    return {}

def get_risk_parameters(risk_level):
    """Get risk parameters based on risk tolerance level"""
    if risk_level == 'low':
        return {
            'max_position_size': 0.05,
            'max_portfolio_risk': 0.01,
            'stop_loss': 0.02,
            'take_profit': 0.03
        }
    elif risk_level == 'high':
        return {
            'max_position_size': 0.2,
            'max_portfolio_risk': 0.03,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    else:  # medium
        return {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02,
            'stop_loss': 0.03,
            'take_profit': 0.05
        }
