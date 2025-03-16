# modules/asset_selector.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from modules.sentiment_analysis import aggregate_sentiments
from config import FMP_API_KEY

def fetch_asset_universe():
    """
    Retrieves S&P 500 tickers from Wikipedia and returns the first 100 for efficiency.
    """
    ticker_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = ticker_df['Symbol'].tolist()
    return tickers[:100]

def calculate_momentum(ticker):
    """
    Computes momentum as percentage change over the last month.
    """
    data = yf.download(ticker, period='1mo', interval='1d')
    if data.empty or len(data['Close']) < 2:
        return -np.inf
    return float((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1)

def fetch_volume(ticker):
    """
    Returns average daily volume over the last 5 days.
    Explicitly converts pandas Series mean to scalar float.
    """
    data = yf.download(ticker, period='5d', interval='1d')
    if data.empty or len(data['Volume']) < 1:
        return 0
    return float(data['Volume'].mean())

def get_sentiment_score(ticker):
    """
    Retrieves recent news sentiment score for the given ticker.
    """
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=5&apikey={FMP_API_KEY}"
    response = requests.get(url).json()
    if not response:
        return 0

    scores = []
    for news_item in response:
        if isinstance(news_item, dict) and 'title' in news_item:
            title = news_item['title']
        else:
            title = str(news_item)
        result = aggregate_sentiments(title)
        score = result.get('score', 0)
        score = float(score)
        score = np.clip(score, -1, 1)
        scores.append(score)

    return float(np.mean(scores))

def score_asset(ticker):
    """
    Generates a composite score combining momentum and sentiment.
    Filters out low liquidity assets explicitly handling potential issues.
    """
    momentum = calculate_momentum(ticker)
    volume = fetch_volume(ticker)
    sentiment = get_sentiment_score(ticker)

    # Ensure all values are scalar floats
    if volume < 1e6 or momentum <= -np.inf:
        return -np.inf

    composite_score = (0.6 * momentum) + (0.4 * sentiment)
    return composite_score

def select_top_assets(n=10):
    """
    Selects the top N assets based on their composite scores.
    """
    tickers = fetch_asset_universe()
    scores = {}
    for ticker in tickers:
        try:
            score = score_asset(ticker)
            if np.isnan(score):
                score = -np.inf
        except Exception as e:
            print(f"Error scoring {ticker}: {e}")
            score = -np.inf

        scores[ticker] = score
        print(f"{ticker} scored {score}")

    sorted_assets = sorted(
        ((ticker, score) for ticker, score in scores.items() if score > -np.inf),
        key=lambda x: x[1],
        reverse=True
    )

    top_assets = [ticker for ticker, score in sorted_assets[:n]]
    return top_assets

if __name__ == "__main__":
    top_assets = select_top_assets(n=10)
    print("Dynamic Watchlist:", top_assets)
