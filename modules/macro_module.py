# modules/macro_module.py
"""
Macroeconomic indicators module for market analysis
"""
import pandas as pd
import numpy as np
import logging
import os
import json
import requests
import fredapi
import time
import datetime
from config import FRED_API_KEY, SystemConfig
from modules.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

# Cache directory
CACHE_DIR = os.path.join(SystemConfig.CACHE_DIR, 'macro')
os.makedirs(CACHE_DIR, exist_ok=True)

class MacroIndicators:
    """Class for retrieving and analyzing macroeconomic indicators"""
    
    def __init__(self, api_key=FRED_API_KEY):
        self.api_key = api_key
        self.fred = self._initialize_fred() if api_key else None
        self.cache_ttl = 86400  # 24 hours in seconds
        
        # Common economic indicators with their FRED codes
        self.indicators = {
            'GDP': 'GDP',                          # Gross Domestic Product
            'GDPC1': 'GDPC1',                      # Real GDP
            'UNRATE': 'UNRATE',                    # Unemployment Rate
            'CPIAUCSL': 'CPIAUCSL',                # Consumer Price Index
            'FEDFUNDS': 'FEDFUNDS',                # Federal Funds Rate
            'T10Y2Y': 'T10Y2Y',                    # 10Y-2Y Treasury Spread
            'T10Y3M': 'T10Y3M',                    # 10Y-3M Treasury Spread
            'INDPRO': 'INDPRO',                    # Industrial Production
            'HOUST': 'HOUST',                      # Housing Starts
            'RETAILSMNSA': 'RETAILSMNSA',          # Retail Sales
            'DCOILWTICO': 'DCOILWTICO',            # WTI Crude Oil Price
            'M2SL': 'M2SL',                        # M2 Money Supply
            'USALOLITONOSTSAM': 'USSLIND',         # Leading Index
            'USREC': 'USREC',                      # Recession Indicator
            'EMRATIO': 'EMRATIO',                  # Employment-Population Ratio
            'DTWEXBGS': 'DTWEXBGS',                # Dollar Index
            'VIXCLS': 'VIXCLS',                    # VIX Index
            'BAMLH0A0HYM2': 'BAMLH0A0HYM2',        # High Yield Bond Spread
            'PSAVERT': 'PSAVERT',                  # Personal Savings Rate
            'UMCSENT': 'UMCSENT'                   # Consumer Sentiment
        }
    
    def _initialize_fred(self):
        """Initialize FRED API client"""
        try:
            from fredapi import Fred
            return Fred(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Error initializing FRED API: {e}")
            return None
    
    def get_cached_series(self, series_id, start_date, end_date):
        """Check if we have cached data for the series"""
        cache_key = f"{series_id}_{start_date}_{end_date}"
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid
            if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        return pd.Series(data['values'], index=pd.to_datetime(data['dates']))
                except Exception as e:
                    logger.warning(f"Error loading cached series: {e}")
        
        return None
    
    def save_cached_series(self, series_id, start_date, end_date, series):
        """Save series to cache"""
        cache_key = f"{series_id}_{start_date}_{end_date}"
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        try:
            # Convert to serializable format
            data = {
                'dates': [d.strftime('%Y-%m-%d') for d in series.index],
                'values': series.values.tolist()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving series to cache: {e}")
    
    def get_series(self, series_id, start_date=None, end_date=None, transform=None, force_refresh=False):
        """
        Get data series from FRED
        
        Parameters:
          - series_id: FRED series ID
          - start_date: Start date (None for 5 years ago)
          - end_date: End date (None for today)
          - transform: Transformation to apply ('pct_change', 'diff', etc.)
          - force_refresh: Force refresh cache
          
        Returns:
          - pd.Series: Data series
        """
        if not self.fred:
            logger.warning("FRED API not initialized, returning empty series")
            return pd.Series()
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_series = self.get_cached_series(series_id, start_date, end_date)
            if cached_series is not None:
                logger.debug(f"Using cached data for {series_id}")
                
                # Apply transformation if requested
                if transform:
                    if transform == 'pct_change':
                        return cached_series.pct_change().dropna()
                    elif transform == 'diff':
                        return cached_series.diff().dropna()
                    elif transform == 'yoy':
                        return cached_series.pct_change(12).dropna()
                
                return cached_series
        
        try:
            # Get data from FRED
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            # Cache the raw data
            self.save_cached_series(series_id, start_date, end_date, series)
            
            # Apply transformation if requested
            if transform:
                if transform == 'pct_change':
                    return series.pct_change().dropna()
                elif transform == 'diff':
                    return series.diff().dropna()
                elif transform == 'yoy':
                    return series.pct_change(12).dropna()
            
            return series
            
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return pd.Series()
    
    def get_multiple_series(self, series_ids, start_date=None, end_date=None, transform=None):
        """
        Get multiple data series from FRED
        
        Parameters:
          - series_ids: List of FRED series IDs
          - start_date: Start date (None for 5 years ago)
          - end_date: End date (None for today)
          - transform: Transformation to apply ('pct_change', 'diff', etc.)
          
        Returns:
          - pd.DataFrame: DataFrame with all series
        """
        data = {}
        
        for series_id in series_ids:
            series = self.get_series(series_id, start_date, end_date, transform)
            if not series.empty:
                data[series_id] = series
        
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    
    def get_indicator(self, indicator_name, start_date=None, end_date=None, transform=None):
        """
        Get a specific economic indicator
        
        Parameters:
          - indicator_name: Name of the indicator (key in self.indicators)
          - start_date: Start date
          - end_date: End date
          - transform: Transformation to apply
          
        Returns:
          - pd.Series: Indicator data
        """
        if indicator_name in self.indicators:
            series_id = self.indicators[indicator_name]
            return self.get_series(series_id, start_date, end_date, transform)
        else:
            logger.warning(f"Unknown indicator: {indicator_name}")
            return pd.Series()
    
    def get_recession_probability(self):
        """
        Calculate recession probability based on yield curve inversion and other indicators
        
        Returns:
          - float: Recession probability (0-1)
        """
        try:
            # Typical recession indicators:
            # 1. Yield curve inversion (10Y-2Y spread)
            # 2. Leading index
            # 3. Unemployment rate trend
            
            # Get latest data
            yield_curve = self.get_series('T10Y2Y', transform=None)
            leading_index = self.get_series('USSLIND', transform=None)
            unemployment = self.get_series('UNRATE', transform='diff')
            
            if yield_curve.empty or leading_index.empty or unemployment.empty:
                logger.warning("Missing data for recession probability calculation")
                return 0.5  # Neutral
            
            # Latest values
            latest_yield_curve = yield_curve.iloc[-1]
            latest_leading_index = leading_index.iloc[-1]
            unemployment_trend = unemployment.iloc[-3:].mean()
            
            # Simple model:
            # - Yield curve: negative = bad
            # - Leading index: negative = bad
            # - Unemployment trend: positive = bad
            
            probability = 0.0
            
            # Yield curve inversion is a strong signal
            if latest_yield_curve < 0:
                probability += 0.4
            elif latest_yield_curve < 0.5:
                probability += 0.2
            
            # Leading index
            if latest_leading_index < 0:
                probability += 0.3
            elif latest_leading_index < 0.3:
                probability += 0.1
            
            # Unemployment trend
            if unemployment_trend > 0.2:
                probability += 0.3
            elif unemployment_trend > 0:
                probability += 0.1
            
            # Cap at 1.0
            return min(probability, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating recession probability: {e}")
            return 0.5  # Neutral
    
    def get_inflation_trend(self):
        """
        Analyze inflation trend
        
        Returns:
          - dict: Inflation trend analysis
        """
        try:
            # Get CPI data
            cpi = self.get_series('CPIAUCSL', transform='pct_change')
            
            if cpi.empty:
                logger.warning("Missing CPI data for inflation trend analysis")
                return {'trend': 'NEUTRAL', 'latest': 0, 'trend_strength': 0}
            
            # Annual inflation rate (latest)
            latest_annual = cpi.iloc[-12:].sum() * 100
            
            # Recent months trend (last 3 months vs previous 3 months)
            recent_trend = cpi.iloc[-3:].mean() - cpi.iloc[-6:-3].mean()
            recent_trend *= 100  # Convert to percentage
            
            # Determine trend
            if recent_trend < -0.2:
                trend = 'DECREASING'
                trend_strength = min(abs(recent_trend) / 0.5, 1.0)
            elif recent_trend > 0.2:
                trend = 'INCREASING'
                trend_strength = min(abs(recent_trend) / 0.5, 1.0)
            else:
                trend = 'STABLE'
                trend_strength = 0.0
            
            return {
                'trend': trend,
                'latest': latest_annual,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing inflation trend: {e}")
            return {'trend': 'NEUTRAL', 'latest': 0, 'trend_strength': 0}
    
    def get_economic_climate(self):
        """
        Analyze overall economic climate
        
        Returns:
          - str: Economic climate ('EXPANSION', 'CONTRACTION', 'STAGFLATION', 'RECOVERY', 'STABLE')
        """
        try:
            # Get key indicators
            gdp_growth = self.get_series('GDPC1', transform='pct_change')
            unemployment = self.get_series('UNRATE', transform=None)
            inflation = self.get_series('CPIAUCSL', transform='pct_change')
            
            if gdp_growth.empty or unemployment.empty or inflation.empty:
                logger.warning("Missing data for economic climate analysis")
                return 'UNKNOWN'
            
            # Latest values
            latest_gdp_growth = gdp_growth.iloc[-1] * 100  # To percentage
            latest_unemployment = unemployment.iloc[-1]
            latest_inflation = inflation.iloc[-12:].sum() * 100  # Annual inflation
            
            # Trend (latest quarter vs previous quarter)
            gdp_trend = latest_gdp_growth - gdp_growth.iloc[-2] * 100
            unemp_trend = unemployment.iloc[-1] - unemployment.iloc[-4]  # 3-month change
            
            # Economic climate determination
            if latest_gdp_growth > 2 and unemp_trend < 0:
                if latest_inflation > 3:
                    return 'EXPANSION_HIGH_INFLATION'
                else:
                    return 'EXPANSION'
            elif latest_gdp_growth < 0.5 and unemp_trend > 0:
                if latest_inflation > 3:
                    return 'STAGFLATION'
                else:
                    return 'CONTRACTION'
            elif latest_gdp_growth > 1 and unemp_trend < 0:
                return 'RECOVERY'
            else:
                return 'STABLE'
            
        except Exception as e:
            logger.error(f"Error analyzing economic climate: {e}")
            return 'UNKNOWN'
    
    def get_market_regime(self):
        """
        Determine current market regime based on economic indicators
        
        Returns:
          - dict: Market regime analysis
        """
        try:
            # Get economic climate
            climate = self.get_economic_climate()
            
            # Get recession probability
            recession_prob = self.get_recession_probability()
            
            # Get interest rate trend
            fed_funds = self.get_series('FEDFUNDS', transform=None)
            if not fed_funds.empty:
                rate_trend = fed_funds.iloc[-1] - fed_funds.iloc[-4]  # 3-month change
            else:
                rate_trend = 0
            
            # Get market volatility
            vix = self.get_series('VIXCLS', transform=None)
            if not vix.empty:
                volatility = vix.iloc[-1]
                volatility_regime = 'HIGH' if volatility > 25 else 'NORMAL' if volatility > 15 else 'LOW'
            else:
                volatility = 0
                volatility_regime = 'UNKNOWN'
            
            # Market regime determination
            regime = {}
            
            # Determine risk-on/risk-off based on economic factors
            if climate in ['EXPANSION', 'RECOVERY'] and recession_prob < 0.3 and rate_trend <= 0:
                regime['risk'] = 'RISK_ON'
            elif climate in ['CONTRACTION', 'STAGFLATION'] or recession_prob > 0.6 or rate_trend > 0.5:
                regime['risk'] = 'RISK_OFF'
            else:
                regime['risk'] = 'NEUTRAL'
            
            # Determine growth/value rotation
            if climate in ['EXPANSION', 'RECOVERY'] and volatility_regime != 'HIGH':
                regime['style'] = 'GROWTH'
            elif climate in ['CONTRACTION', 'STAGFLATION'] or volatility_regime == 'HIGH':
                regime['style'] = 'VALUE'
            else:
                regime['style'] = 'BALANCED'
            
            # Determine sector rotation
            if climate == 'EXPANSION':
                regime['sectors'] = ['TECH', 'CONSUMER_DISCRETIONARY', 'INDUSTRIALS']
            elif climate == 'EXPANSION_HIGH_INFLATION':
                regime['sectors'] = ['ENERGY', 'MATERIALS', 'FINANCIALS']
            elif climate == 'CONTRACTION':
                regime['sectors'] = ['UTILITIES', 'HEALTHCARE', 'CONSUMER_STAPLES']
            elif climate == 'STAGFLATION':
                regime['sectors'] = ['ENERGY', 'MATERIALS', 'UTILITIES']
            elif climate == 'RECOVERY':
                regime['sectors'] = ['FINANCIALS', 'CONSUMER_DISCRETIONARY', 'REAL_ESTATE']
            else:
                regime['sectors'] = ['BALANCED']
            
            # Add additional info
            regime['climate'] = climate
            regime['recession_probability'] = recession_prob
            regime['volatility_regime'] = volatility_regime
            
            return regime
            
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return {'risk': 'NEUTRAL', 'style': 'BALANCED', 'sectors': ['BALANCED'], 'climate': 'UNKNOWN'}

def get_gdp_indicator():
    """
    Get simple GDP-based market indicator (for use in signal aggregation)
    
    Returns:
      - str: Market indicator ('BULLISH', 'BEARISH', 'NEUTRAL')
    """
    # Initialize macro indicators
    macro = MacroIndicators()
    
    try:
        # Get GDP data
        gdp = macro.get_series('GDPC1', transform='pct_change')
        
        if gdp.empty:
            logger.warning("No GDP data available")
            return "NEUTRAL"
        
        # Latest GDP growth (annualized)
        latest_growth = gdp.iloc[-1] * 400  # Quarterly to annual
        
        # Previous GDP growth
        prev_growth = gdp.iloc[-2] * 400
        
        # Determine indicator
        if latest_growth > 2.5 or (latest_growth > 1.0 and latest_growth > prev_growth):
            return "BULLISH"
        elif latest_growth < 0 or (latest_growth < 1.0 and latest_growth < prev_growth):
            return "BEARISH"
        else:
            return "NEUTRAL"
        
    except Exception as e:
        logger.error(f"Error getting GDP indicator: {e}")
        return "NEUTRAL"

def get_market_status():
    """
    Get comprehensive market status based on multiple indicators
    
    Returns:
      - dict: Market status
    """
    # Initialize macro indicators
    macro = MacroIndicators()
    
    try:
        # Get market regime
        regime = macro.get_market_regime()
        
        # Get inflation trend
        inflation = macro.get_inflation_trend()
        
        # Get recession probability
        recession_prob = macro.get_recession_probability()
        
        # Create market status
        status = {
            'market_regime': regime,
            'inflation': inflation,
            'recession_probability': recession_prob,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to cache for dashboard use
        status_file = os.path.join(CACHE_DIR, 'market_status.json')
        with open(status_file, 'w') as f:
            json.dump(status, f, default=str)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return {
            'market_regime': {'risk': 'NEUTRAL', 'style': 'BALANCED'},
            'inflation': {'trend': 'NEUTRAL'},
            'recession_probability': 0.5,
            'timestamp': datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize macro indicators
    macro = MacroIndicators()
    
    # Test getting a series
    gdp = macro.get_series('GDP', transform=None)
    print(f"GDP data points: {len(gdp)}")
    print(f"Latest GDP: {gdp.iloc[-1]}")
    
    # Test getting multiple series
    multiple_series = macro.get_multiple_series(['UNRATE', 'CPIAUCSL', 'FEDFUNDS'])
    print(f"Multiple series shape: {multiple_series.shape}")
    print(f"Latest values:\n{multiple_series.iloc[-1]}")
    
    # Test recession probability
    recession_prob = macro.get_recession_probability()
    print(f"Recession probability: {recession_prob:.2f}")
    
    # Test inflation trend
    inflation_trend = macro.get_inflation_trend()
    print(f"Inflation trend: {inflation_trend['trend']}, Latest: {inflation_trend['latest']:.2f}%")
    
    # Test economic climate
    climate = macro.get_economic_climate()
    print(f"Economic climate: {climate}")
    
    # Test market regime
    regime = macro.get_market_regime()
    print(f"Market regime: {regime}")
    
    # Test GDP indicator
    gdp_indicator = get_gdp_indicator()
    print(f"GDP indicator: {gdp_indicator}")
    
    # Test comprehensive market status
    status = get_market_status()
    print(f"Market status: {json.dumps(status, indent=2, default=str)}")