# modules/news_social_monitor.py
"""
News and social media monitoring module for gathering market sentiment data
"""
import praw
import requests
import json
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from config import (
    NEWS_API_KEY, 
    REDDIT_CLIENT_ID, 
    REDDIT_CLIENT_SECRET, 
    REDDIT_USER_AGENT,
    SystemConfig
)
from modules.sentiment_analysis import analyze_sentiment, aggregate_sentiments
from modules.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

# Cache directory
CACHE_DIR = os.path.join(SystemConfig.CACHE_DIR, 'news_social')
os.makedirs(CACHE_DIR, exist_ok=True)

class NewsMonitor:
    """News monitoring class for retrieving and analyzing financial news"""
    
    def __init__(self, api_key=NEWS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.join(CACHE_DIR, 'news'), exist_ok=True)
    
    def get_cached_news(self, query, days=1):
        """Check if we have cached news for the query"""
        cache_key = query.replace(' ', '_').lower()
        cache_file = os.path.join(CACHE_DIR, 'news', f"{cache_key}_{days}d.json")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid
            if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading cached news: {e}")
        
        return None
    
    def save_cached_news(self, query, days, data):
        """Save news data to cache"""
        cache_key = query.replace(' ', '_').lower()
        cache_file = os.path.join(CACHE_DIR, 'news', f"{cache_key}_{days}d.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving news to cache: {e}")
    
    def get_news(self, query, days=1, force_refresh=False):
        """
        Get news articles for the given query
        
        Parameters:
          - query: Search query (e.g., "AAPL stock")
          - days: Number of days to look back
          - force_refresh: Force refresh cache
          
        Returns:
          - list: List of news articles
        """
        if not self.api_key:
            logger.warning("News API key not configured, returning empty results")
            return []
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.get_cached_news(query, days)
            if cached_data:
                logger.debug(f"Using cached news for {query}")
                return cached_data
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format dates for API
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # Build URL
        url = f"{self.base_url}/everything"
        
        # Build parameters
        params = {
            'q': query,
            'from': from_date_str,
            'to': to_date_str,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 25,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code != 200:
                logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                }
                processed_articles.append(processed_article)
            
            # Cache results
            self.save_cached_news(query, days, processed_articles)
            
            logger.info(f"Retrieved {len(processed_articles)} news articles for {query}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_news_sentiment(self, query, days=1, method="hybrid", symbol=None):
        """
        Get and analyze sentiment of news articles
        
        Parameters:
          - query: Search query
          - days: Number of days to look back
          - method: Sentiment analysis method
          - symbol: Optional stock symbol
          
        Returns:
          - dict: Aggregated sentiment result
        """
        articles = self.get_news(query, days)
        
        if not articles:
            logger.warning(f"No news articles found for {query}")
            return {
                "score": 0,
                "label": "NEUTRAL",
                "details": {"error": "No news articles found"},
                "method": method,
                "count": 0
            }
        
        # Extract text from articles
        texts = []
        for article in articles:
            # Combine title and description for better context
            text = f"{article['title']}. {article['description']}"
            texts.append(text)
        
        # Analyze sentiment
        result = aggregate_sentiments(texts, method=method, symbol=symbol)
        
        # Add article count
        result["count"] = len(articles)
        
        # Add sources
        result["sources"] = [article["source"] for article in articles[:5]]
        
        return result

class SocialMediaMonitor:
    """Social media monitoring class for retrieving and analyzing social media posts"""
    
    def __init__(self, client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = self._initialize_reddit() if client_id and client_secret else None
        self.cache_ttl = 1800  # 30 minutes in seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.join(CACHE_DIR, 'social'), exist_ok=True)
    
    def _initialize_reddit(self):
        """Initialize Reddit API client"""
        try:
            return praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
        except Exception as e:
            logger.error(f"Error initializing Reddit API: {e}")
            return None
    
    def get_cached_posts(self, query, subreddit, limit):
        """Check if we have cached posts for the query"""
        cache_key = f"{query.replace(' ', '_').lower()}_{subreddit}_{limit}"
        cache_file = os.path.join(CACHE_DIR, 'social', f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid
            if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading cached posts: {e}")
        
        return None
    
    def save_cached_posts(self, query, subreddit, limit, data):
        """Save posts to cache"""
        cache_key = f"{query.replace(' ', '_').lower()}_{subreddit}_{limit}"
        cache_file = os.path.join(CACHE_DIR, 'social', f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving posts to cache: {e}")
    
    def get_reddit_posts(self, query, subreddit="stocks", limit=25, force_refresh=False):
        """
        Get Reddit posts for the given query
        
        Parameters:
          - query: Search query
          - subreddit: Subreddit to search in
          - limit: Maximum number of posts to retrieve
          - force_refresh: Force refresh cache
          
        Returns:
          - list: List of Reddit posts
        """
        if not self.reddit:
            logger.warning("Reddit API not configured, returning empty results")
            return []
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.get_cached_posts(query, subreddit, limit)
            if cached_data:
                logger.debug(f"Using cached Reddit posts for {query} in r/{subreddit}")
                return cached_data
        
        try:
            # Get subreddit
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Search for posts
            posts = list(subreddit_obj.search(query, limit=limit, sort='new'))
            
            # Process posts
            processed_posts = []
            for post in posts:
                # Get top comments
                post.comments.replace_more(limit=0)
                top_comments = [comment.body for comment in post.comments.list()[:5]]
                
                processed_post = {
                    'title': post.title,
                    'selftext': post.selftext,
                    'url': f"https://www.reddit.com{post.permalink}",
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'top_comments': top_comments
                }
                processed_posts.append(processed_post)
            
            # Cache results
            self.save_cached_posts(query, subreddit, limit, processed_posts)
            
            logger.info(f"Retrieved {len(processed_posts)} Reddit posts for {query} in r/{subreddit}")
            return processed_posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    def get_reddit_sentiment(self, query, subreddit="stocks", limit=25, method="vader", symbol=None):
        """
        Get and analyze sentiment of Reddit posts and comments
        
        Parameters:
          - query: Search query
          - subreddit: Subreddit to search in
          - limit: Maximum number of posts to retrieve
          - method: Sentiment analysis method
          - symbol: Optional stock symbol
          
        Returns:
          - dict: Aggregated sentiment result
        """
        posts = self.get_reddit_posts(query, subreddit, limit)
        
        if not posts:
            logger.warning(f"No Reddit posts found for {query} in r/{subreddit}")
            return {
                "score": 0,
                "label": "NEUTRAL",
                "details": {"error": "No Reddit posts found"},
                "method": method,
                "count": 0
            }
        
        # Extract text from posts and comments
        texts = []
        for post in posts:
            # Combine title and text for better context
            post_text = f"{post['title']}. {post['selftext']}"
            texts.append(post_text)
            
            # Add top comments
            texts.extend(post['top_comments'])
        
        # Remove empty texts
        texts = [text for text in texts if text and len(text.strip()) > 0]
        
        # Analyze sentiment
        result = aggregate_sentiments(texts, method=method, symbol=symbol)
        
        # Add post count
        result["count"] = len(posts)
        result["comment_count"] = sum(len(post['top_comments']) for post in posts)
        
        return result

def get_combined_sentiment(query, days=3, method="hybrid", symbol=None):
    """
    Get combined sentiment from news and social media
    
    Parameters:
      - query: Search query
      - days: Number of days to look back
      - method: Sentiment analysis method
      - symbol: Optional stock symbol
      
    Returns:
      - dict: Combined sentiment result
    """
    news_monitor = NewsMonitor()
    social_monitor = SocialMediaMonitor()
    
    # Get news sentiment
    news_sentiment = news_monitor.analyze_news_sentiment(query, days, method, symbol)
    
    # Get Reddit sentiment
    reddit_sentiment = social_monitor.get_reddit_sentiment(query, "stocks", 25, method, symbol)
    
    # Get additional Reddit sentiment from more specific subreddits
    if symbol:
        # For stocks, check wallstreetbets and investing
        wsb_sentiment = social_monitor.get_reddit_sentiment(symbol, "wallstreetbets", 15, method, symbol)
        investing_sentiment = social_monitor.get_reddit_sentiment(symbol, "investing", 15, method, symbol)
        
        # Weight sentiments based on post count
        weights = [
            news_sentiment["count"] * 2,  # News gets double weight
            reddit_sentiment["count"],
            wsb_sentiment["count"],
            investing_sentiment["count"]
        ]
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [0.4, 0.2, 0.2, 0.2]  # Default weights
        
        # Calculate weighted score
        combined_score = (
            news_sentiment["score"] * weights[0] +
            reddit_sentiment["score"] * weights[1] +
            wsb_sentiment["score"] * weights[2] +
            investing_sentiment["score"] * weights[3]
        )
    else:
        # For market sentiment, just combine news and general Reddit
        weights = [0.7, 0.3]  # News gets more weight for general market
        combined_score = news_sentiment["score"] * weights[0] + reddit_sentiment["score"] * weights[1]
    
    # Determine label
    if combined_score >= 0.05:
        label = "POSITIVE"
    elif combined_score <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    # Create result
    result = {
        "score": combined_score,
        "label": label,
        "details": {
            "news": {
                "score": news_sentiment["score"],
                "label": news_sentiment["label"],
                "count": news_sentiment["count"]
            },
            "reddit": {
                "score": reddit_sentiment["score"],
                "label": reddit_sentiment["label"],
                "count": reddit_sentiment["count"]
            }
        },
        "method": method,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add symbol-specific Reddit sentiment if available
    if symbol:
        result["details"]["wallstreetbets"] = {
            "score": wsb_sentiment["score"],
            "label": wsb_sentiment["label"],
            "count": wsb_sentiment["count"]
        }
        result["details"]["investing"] = {
            "score": investing_sentiment["score"],
            "label": investing_sentiment["label"],
            "count": investing_sentiment["count"]
        }
    
    return result

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test news monitoring
    news_monitor = NewsMonitor()
    news = news_monitor.get_news("AAPL stock", days=3)
    print(f"Found {len(news)} news articles")
    
    news_sentiment = news_monitor.analyze_news_sentiment("AAPL stock", days=3, symbol="AAPL")
    print(f"News sentiment: {news_sentiment['score']:.2f} ({news_sentiment['label']})")
    
    # Test social media monitoring
    social_monitor = SocialMediaMonitor()
    posts = social_monitor.get_reddit_posts("AAPL", "stocks", 10)
    print(f"Found {len(posts)} Reddit posts")
    
    reddit_sentiment = social_monitor.get_reddit_sentiment("AAPL", "stocks", 10, symbol="AAPL")
    print(f"Reddit sentiment: {reddit_sentiment['score']:.2f} ({reddit_sentiment['label']})")
    
    # Test combined sentiment
    combined = get_combined_sentiment("AAPL stock", days=3, symbol="AAPL")
    print(f"Combined sentiment: {combined['score']:.2f} ({combined['label']})")