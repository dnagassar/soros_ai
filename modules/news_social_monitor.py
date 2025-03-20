# modules/news_social_monitor.py
import requests
import feedparser
import praw
from datetime import datetime, timedelta
from modules.sentiment_analysis import aggregate_sentiments
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, NEWS_API_KEY

def fetch_reddit_data(query, subreddit="stocks", limit=50):
    """
    Fetches Reddit post titles from the specified subreddit that match the query.
    """
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)
    # Search posts in the specified subreddit that match the query.
    posts = reddit.subreddit(subreddit).search(query, limit=limit)
    titles = [post.title for post in posts if post.title]
    return titles

def fetch_stocktwits_data(symbol):
    """
    Fetches messages from StockTwits for the given symbol.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        messages = [msg.get("body") for msg in data.get("messages", []) if msg.get("body")]
        return messages
    else:
        print("Error fetching StockTwits data:", response.text)
        return []

def fetch_rss_data(query):
    """
    Fetches news headlines from Google News RSS feed for the given query.
    Requires the feedparser package.
    """
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    headlines = [entry.title for entry in feed.entries if entry.title]
    return headlines

def get_combined_sentiment(query, symbol=None):
    """
    Aggregates sentiment from Reddit, StockTwits, and Google News RSS feeds.
    - `query` is used for Reddit search and RSS feeds.
    - `symbol` is used for StockTwits; if None, StockTwits is skipped.
    Returns a dictionary with the combined sentiment label and score.
    """
    reddit_texts = fetch_reddit_data(query)
    stocktwits_texts = fetch_stocktwits_data(symbol) if symbol else []
    rss_texts = fetch_rss_data(query)
    
    # Combine texts from all sources
    combined_texts = reddit_texts + stocktwits_texts + rss_texts
    # Filter out any None or empty strings.
    combined_texts = [text for text in combined_texts if isinstance(text, str) and text.strip()]
    
    if not combined_texts:
        return {"sentiment": "NEUTRAL", "score": 0}
    
    # Compute sentiment for each text using your existing sentiment aggregator.
    scores = [aggregate_sentiments(text).get("score", 0) for text in combined_texts]
    avg_score = sum(scores) / len(scores)
    sentiment = "POSITIVE" if avg_score > 0.5 else "NEGATIVE" if avg_score < -0.5 else "NEUTRAL"
    return {"sentiment": sentiment, "score": avg_score}

if __name__ == "__main__":
    # Example usage: Get combined sentiment for AAPL using all sources.
    result = get_combined_sentiment("AAPL", symbol="AAPL")
    print("Combined Reddit/StockTwits/RSS Sentiment for AAPL:", result)
