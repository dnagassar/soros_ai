# modules/asset_selector.py
import numpy as np
import pandas as pd
import requests
import logging
import os
from datetime import datetime, timedelta
import json
from functools import lru_cache

from config import FMP_API_KEY, TIINGO_API_KEY, QUANDL_API_KEY, SystemConfig
from modules.sentiment_analysis import aggregate_sentiments
from modules.news_social_monitor import get_combined_sentiment
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data

# Configure logging
logger = logging.getLogger(__name__)

class AssetSelector:
    """
    Enhanced asset selector with multiple data sources, caching, and
    configurable selection criteria
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize the asset selector with optional caching
        
        Parameters:
          - cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir or SystemConfig.CACHE_DIR
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Default selection criteria weights
        self.criteria_weights = {
            'momentum': 0.4,
            'volatility': 0.2,
            'volume': 0.1,
            'sentiment': 0.3
        }
        
        # Default universe sources in order of preference
        self.universe_sources = [
            'sp500_wiki',
            'russell2000',
            'nasdaq100',