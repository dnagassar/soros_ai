# modules/asset_selector.py
"""
Asset Selector Module for selecting assets to trade based on various criteria
"""
import pandas as pd
import numpy as np
import logging
import os
import json
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pickle
from config import ALPHA_VANTAGE_API_KEY, FMP_API_KEY, SystemConfig
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data
from modules.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

# Cache directory
CACHE_DIR = os.path.join(SystemConfig.CACHE_DIR, 'asset_selector')
os.makedirs(CACHE_DIR, exist_ok=True)

# Default market assets by sector
DEFAULT_ASSETS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CSCO", "ORCL"],
    "Healthcare": ["JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY", "TMO", "ABT", "BMY"],
    "Financial": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP"],
    "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "SBUX", "NKE", "COST"],
    "Industrial": ["CAT", "BA", "GE", "MMM", "HON", "UNP", "UPS", "LMT", "RTX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "XEL", "EXC", "SRE"],
    "Materials": ["LIN", "APD", "ECL", "DD", "NEM", "FCX", "NUE"],
    "Communication": ["T", "VZ", "CMCSA", "NFLX", "DIS", "TMUS"],
    "Real Estate": ["AMT", "PLD", "CCI", "SPG", "EQIX", "O", "AVB"]
}

def get_sp500_components():
    """
    Get S&P 500 components from Wikipedia or cached data
    
    Returns:
      - pd.DataFrame: DataFrame with S&P 500 components
    """
    cache_file = os.path.join(CACHE_DIR, 'sp500_components.pickle')
    
    # Check if cache exists and is fresh (less than 7 days old)
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - cache_time).days < 7:
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading S&P 500 cache: {e}")
    
    # Fetch from Wikipedia
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        
        # Clean up column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f"Retrieved S&P 500 components: {len(df)} companies")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching S&P 500 components: {e}")
        
        # Return default sectors if available
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        # Create a dummy DataFrame as last resort
        symbols = []
        sectors = []
        for sector, tickers in DEFAULT_ASSETS.items():
            for ticker in tickers:
                symbols.append(ticker)
                sectors.append(sector)
        
        dummy_df = pd.DataFrame({
            'Symbol': symbols,
            'GICS_Sector': sectors
        })
        
        return dummy_df

def get_nasdaq100_components():
    """
    Get Nasdaq 100 components from Wikipedia or cached data
    
    Returns:
      - pd.DataFrame: DataFrame with Nasdaq 100 components
    """
    cache_file = os.path.join(CACHE_DIR, 'nasdaq100_components.pickle')
    
    # Check if cache exists and is fresh (less than 7 days old)
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - cache_time).days < 7:
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading Nasdaq 100 cache: {e}")
    
    # Fetch from Wikipedia
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]  # The table with the components
        
        # Clean up
        df.columns = ['Company', 'Ticker', 'GICS_Sector', 'GICS_Sub_Industry', 'Founded', 'Added']
        
        # Extract just the ticker symbol (sometimes has notes)
        df['Symbol'] = df['Ticker'].str.split(expand=True)[0]
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f"Retrieved Nasdaq 100 components: {len(df)} companies")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching Nasdaq 100 components: {e}")
        
        # Return default tech stocks if available
        tech_stocks = DEFAULT_ASSETS.get("Technology", [])
        dummy_df = pd.DataFrame({
            'Symbol': tech_stocks,
            'GICS_Sector': ['Technology'] * len(tech_stocks)
        })
        
        return dummy_df

def get_market_cap_data(symbols):
    """
    Get market cap data for the given symbols
    
    Parameters:
      - symbols: List of symbols
      
    Returns:
      - dict: Dictionary with market cap data
    """
    cache_file = os.path.join(CACHE_DIR, 'market_cap.pickle')
    
    # Check if cache exists and is fresh (less than 1 day old)
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - cache_time).total_seconds() < 86400:  # 24 hours
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                    # Check if we have data for all symbols
                    missing_symbols = [s for s in symbols if s not in cached_data]
                    if not missing_symbols:
                        return cached_data
                    
                    # Filter for requested symbols
                    market_caps = {s: cached_data[s] for s in symbols if s in cached_data}
                    
                    # Add the missing symbols below
                    symbols = missing_symbols
            except Exception as e:
                logger.warning(f"Error loading market cap cache: {e}")
                market_caps = {}
        else:
            market_caps = {}
    else:
        market_caps = {}
    
    # Try to get market cap data from FMP API
    if FMP_API_KEY:
        try:
            # Split into batches of 20 symbols
            batch_size = 20
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i+batch_size]
                symbols_str = ','.join(batch_symbols)
                
                url = f"https://financialmodelingprep.com/api/v3/market-capitalization/{symbols_str}?apikey={FMP_API_KEY}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        symbol = item.get('symbol')
                        if symbol:
                            market_caps[symbol] = item.get('marketCap', 0)
                
                # Respect API rate limits
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error fetching market cap data from FMP: {e}")
    
    # For symbols that are still missing, try yfinance as fallback
    missing_symbols = [s for s in symbols if s not in market_caps]
    if missing_symbols:
        try:
            for symbol in missing_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    market_caps[symbol] = info.get('marketCap', 0)
                except Exception:
                    # Set to 0 if we can't get the data
                    market_caps[symbol] = 0
                
                # Respect API rate limits
                time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error fetching market cap data from yfinance: {e}")
    
    # Cache the data
    try:
        # Load existing cache if it exists
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                existing_cache = pickle.load(f)
                # Update with new data
                existing_cache.update(market_caps)
                market_caps = existing_cache
        
        with open(cache_file, 'wb') as f:
            pickle.dump(market_caps, f)
    except Exception as e:
        logger.warning(f"Error caching market cap data: {e}")
    
    return market_caps

def calculate_momentum(price_data, lookback_periods=[20, 60, 120]):
    """
    Calculate momentum for the given price data
    
    Parameters:
      - price_data: Dictionary of price DataFrames by symbol
      - lookback_periods: List of lookback periods in days
      
    Returns:
      - dict: Dictionary with momentum scores
    """
    momentum_scores = {}
    
    for symbol, df in price_data.items():
        if df is None or df.empty:
            momentum_scores[symbol] = 0
            continue
        
        # Calculate returns for each lookback period
        returns = {}
        for period in lookback_periods:
            if len(df) > period:
                # Use adjusted close if available, otherwise close
                if 'Adj Close' in df.columns:
                    price_col = 'Adj Close'
                else:
                    price_col = 'Close'
                
                returns[period] = df[price_col].iloc[-1] / df[price_col].iloc[-period-1] - 1
            else:
                returns[period] = 0
        
        # Calculate weighted momentum score
        # Shorter periods get higher weight
        weights = [3, 2, 1]  # Weights for 20, 60, 120 days
        
        # Ensure we have the same number of weights as periods
        weights = weights[:len(lookback_periods)]
        
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        # Calculate score
        score = sum(returns[period] * weight for period, weight in zip(lookback_periods, weights))
        
        momentum_scores[symbol] = score
    
    return momentum_scores

def calculate_volatility(price_data, lookback_period=60):
    """
    Calculate volatility for the given price data
    
    Parameters:
      - price_data: Dictionary of price DataFrames by symbol
      - lookback_period: Lookback period in days
      
    Returns:
      - dict: Dictionary with volatility scores
    """
    volatility_scores = {}
    
    for symbol, df in price_data.items():
        if df is None or df.empty or len(df) < lookback_period:
            volatility_scores[symbol] = 0
            continue
        
        # Calculate daily returns
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        else:
            price_col = 'Close'
        
        returns = df[price_col].pct_change().dropna()
        
        # Calculate annualized volatility (standard deviation of returns)
        volatility = returns.tail(lookback_period).std() * (252 ** 0.5)  # Annualized
        
        volatility_scores[symbol] = volatility
    
    return volatility_scores

def create_composite_score(symbols, market_caps, momentum_scores, volatility_scores, 
                          weights={'market_cap': 0.3, 'momentum': 0.5, 'volatility': 0.2}):
    """
    Create a composite score for asset selection
    
    Parameters:
      - symbols: List of symbols
      - market_caps: Dictionary of market caps
      - momentum_scores: Dictionary of momentum scores
      - volatility_scores: Dictionary of volatility scores
      - weights: Dictionary of weights for each factor
      
    Returns:
      - pd.DataFrame: DataFrame with composite scores
    """
    # Create DataFrame
    data = []
    for symbol in symbols:
        market_cap = market_caps.get(symbol, 0)
        momentum = momentum_scores.get(symbol, 0)
        volatility = volatility_scores.get(symbol, 0)
        
        data.append({
            'Symbol': symbol,
            'Market_Cap': market_cap,
            'Momentum': momentum,
            'Volatility': volatility
        })
    
    df = pd.DataFrame(data)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Normalize values to 0-1 range
    for col in ['Market_Cap', 'Momentum', 'Volatility']:
        if df[col].max() - df[col].min() > 0:
            df[f'{col}_Normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[f'{col}_Normalized'] = 0
    
    # Invert volatility (lower is better)
    df['Volatility_Normalized'] = 1 - df['Volatility_Normalized']
    
    # Calculate composite score
    df['Composite_Score'] = (
        df['Market_Cap_Normalized'] * weights['market_cap'] +
        df['Momentum_Normalized'] * weights['momentum'] +
        df['Volatility_Normalized'] * weights['volatility']
    )
    
    # Sort by composite score
    df.sort_values('Composite_Score', ascending=False, inplace=True)
    
    return df

def select_assets_by_sector(df, sector_weights=None, n=10):
    """
    Select assets by sector with weights
    
    Parameters:
      - df: DataFrame with asset data including sector
      - sector_weights: Dictionary of sector weights
      - n: Total number of assets to select
      
    Returns:
      - list: List of selected assets
    """
    # Ensure we have a sector column
    if 'GICS_Sector' not in df.columns:
        logger.warning("No sector information available, selecting top assets overall")
        return df.head(n)['Symbol'].tolist()
    
    # Default sector weights (equal weight if not provided)
    if sector_weights is None:
        sectors = df['GICS_Sector'].unique()
        sector_weights = {sector: 1 / len(sectors) for sector in sectors}
    
    # Calculate number of assets to select from each sector
    sector_counts = {}
    remaining = n
    
    for sector, weight in sector_weights.items():
        count = max(1, int(n * weight))
        sector_counts[sector] = count
        remaining -= count
    
    # Distribute remaining slots to largest sectors
    if remaining > 0:
        sectors_by_size = df.groupby('GICS_Sector').size().sort_values(ascending=False)
        for sector in sectors_by_size.index:
            if sector in sector_counts and remaining > 0:
                sector_counts[sector] += 1
                remaining -= 1
                if remaining == 0:
                    break
    
    # Select top assets from each sector
    selected_assets = []
    
    for sector, count in sector_counts.items():
        sector_df = df[df['GICS_Sector'] == sector]
        if not sector_df.empty:
            # Add top assets from this sector
            sector_assets = sector_df.head(count)['Symbol'].tolist()
            selected_assets.extend(sector_assets)
    
    # If we don't have enough assets, fill with top overall
    if len(selected_assets) < n:
        remaining = n - len(selected_assets)
        # Exclude already selected assets
        remaining_df = df[~df['Symbol'].isin(selected_assets)]
        additional_assets = remaining_df.head(remaining)['Symbol'].tolist()
        selected_assets.extend(additional_assets)
    
    return selected_assets

def select_top_assets(universe="sp500", n=10, lookback_days=120, 
                     sector_balanced=True, sector_weights=None,
                     factor_weights={'market_cap': 0.3, 'momentum': 0.5, 'volatility': 0.2}):
    """
    Select top assets based on multiple factors
    
    Parameters:
      - universe: Asset universe ('sp500', 'nasdaq100', or list of symbols)
      - n: Number of assets to select
      - lookback_days: Lookback period for calculations
      - sector_balanced: Whether to balance by sector
      - sector_weights: Dictionary of sector weights
      - factor_weights: Dictionary of factor weights
      
    Returns:
      - list: List of selected assets
    """
    logger.info(f"Selecting {n} assets from {universe} universe")
    
    # Get universe of assets
    if universe == "sp500":
        df = get_sp500_components()
        symbols = df['Symbol'].tolist()
    elif universe == "nasdaq100":
        df = get_nasdaq100_components()
        symbols = df['Symbol'].tolist()
    elif isinstance(universe, list):
        symbols = universe
        df = pd.DataFrame({'Symbol': symbols})
    else:
        logger.warning(f"Unknown universe: {universe}, using default assets")
        # Flatten the default assets
        symbols = []
        for sector_assets in DEFAULT_ASSETS.values():
            symbols.extend(sector_assets)
        df = pd.DataFrame({'Symbol': symbols})
    
    # Fetch price data for all symbols
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching price data for {len(symbols)} symbols from {start_date} to {end_date}")
    price_data = batch_fetch_price_data(symbols, start_date, end_date, batch_size=20)
    
    # Get market cap data
    market_caps = get_market_cap_data(symbols)
    
    # Calculate momentum scores
    momentum_scores = calculate_momentum(price_data)
    
    # Calculate volatility scores
    volatility_scores = calculate_volatility(price_data)
    
    # Create composite score
    score_df = create_composite_score(
        symbols, market_caps, momentum_scores, volatility_scores, factor_weights
    )
    
    # Merge with universe data if available
    if 'GICS_Sector' in df.columns:
        score_df = pd.merge(score_df, df[['Symbol', 'GICS_Sector']], on='Symbol', how='left')
    
    # Select assets
    if sector_balanced and 'GICS_Sector' in score_df.columns:
        selected_assets = select_assets_by_sector(score_df, sector_weights, n)
    else:
        selected_assets = score_df.head(n)['Symbol'].tolist()
    
    logger.info(f"Selected {len(selected_assets)} assets: {selected_assets}")
    
    return selected_assets

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test S&P 500 components
    sp500 = get_sp500_components()
    print(f"S&P 500 components: {len(sp500)}")
    print(sp500.head())
    
    # Test Nasdaq 100 components
    nasdaq100 = get_nasdaq100_components()
    print(f"Nasdaq 100 components: {len(nasdaq100)}")
    print(nasdaq100.head())
    
    # Test asset selection
    assets = select_top_assets(universe="sp500", n=10)
    print(f"Selected assets: {assets}")
    
    # Test sector-balanced selection
    sector_weights = {
        "Technology": 0.3,
        "Healthcare": 0.2,
        "Financial": 0.15,
        "Consumer": 0.15,
        "Industrial": 0.1,
        "Energy": 0.05,
        "Utilities": 0.05
    }
    
    balanced_assets = select_top_assets(
        universe="sp500", 
        n=10, 
        sector_balanced=True,
        sector_weights=sector_weights
    )
    
    print(f"Sector-balanced assets: {balanced_assets}")