# modules/asset_selector.py
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from config import FMP_API_KEY
from modules.sentiment_analysis import aggregate_sentiments
from modules.news_social_monitor import get_combined_sentiment

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex columns if present.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    return df

def fetch_asset_universe(index='^GSPC'):
    try:
        ticker_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = ticker_df['Symbol'].tolist()
        return tickers if tickers else ["AAPL", "MSFT", "GOOG"]
    except Exception as e:
        print("Error fetching asset universe:", e)
        return ["AAPL", "MSFT", "GOOG"]

def calculate_momentum(ticker):
    try:
        data = yf.download(ticker, period='1mo', interval='1d')
        data = flatten_columns(data)
        if data.empty:
            return -np.inf
        momentum = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        return momentum
    except Exception as e:
        print(f"Error calculating momentum for {ticker}: {e}")
        return -np.inf

def calculate_volatility(ticker):
    try:
        data = yf.download(ticker, period='1mo', interval='1d')
        data = flatten_columns(data)
        if data.empty:
            return np.inf
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std()
        return float(volatility)
    except Exception as e:
        print(f"Error calculating volatility for {ticker}: {e}")
        return np.inf

def fetch_volume(ticker):
    try:
        data = yf.download(ticker, period='5d', interval='1d')
        data = flatten_columns(data)
        if data.empty:
            return 0
        return data['Volume'].mean().item()
    except Exception as e:
        print(f"Error fetching volume for {ticker}: {e}")
        return 0

def get_sentiment_score(ticker, use_social=False):
    """
    Returns the sentiment score for a given ticker.
    If use_social is True, it uses combined sentiment from Reddit, StockTwits, and RSS feeds.
    Otherwise, it falls back to a basic sentiment extraction.
    """
    if use_social:
        score_dict = get_combined_sentiment(ticker, symbol=ticker)
        return score_dict.get("score", 0)
    else:
        base_sentiment = aggregate_sentiments("Recent news for " + ticker)
        return base_sentiment.get("score", 0)

def score_asset(ticker, use_social=False):
    momentum = calculate_momentum(ticker)
    volatility = calculate_volatility(ticker)
    volume = fetch_volume(ticker)
    sentiment = get_sentiment_score(ticker, use_social)
    
    if volume < 1e6:
        return -np.inf

    volatility_score = 1 / volatility if volatility > 0 else 0
    weight_momentum = 0.4
    weight_sentiment = 0.4
    weight_volatility = 0.2

    score = (momentum * weight_momentum) + (sentiment * weight_sentiment) + (volatility_score * weight_volatility)
    return score

def select_top_assets(n=10, use_social=False):
    tickers = fetch_asset_universe()
    scores = {}
    for ticker in tickers:
        scores[ticker] = score_asset(ticker, use_social)
    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_assets = [ticker for ticker, score in sorted_assets[:n] if score != -np.inf]
    return top_assets

if __name__ == "__main__":
    top_assets = select_top_assets(n=10, use_social=True)
    print("Dynamic Watchlist:", top_assets)
