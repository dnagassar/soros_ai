# modules/news_social_monitor.py
"""
News and Social Media Monitor Module - Collects and analyzes news and social media data
"""
import requests
import feedparser
import time
import logging
import json
import pandas as pd
import datetime
import os
import pickle
from functools import lru_cache
import re

# Try to import optional dependencies
try:
    import praw  # Reddit API wrapper
except ImportError:
    praw = None

try:
    from newsapi import NewsApiClient  # News API wrapper
except ImportError:
    NewsApiClient = None

from config import (
    NEWS_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    SystemConfig
)

from modules.sentiment_analysis import analyze_sentiment_vader, analyze_sentiment_hf, aggregate_sentiments
from modules.logger import get_module_logger

# Configure logger
logger = get_module_logger('news_social_monitor')

class NewsSocialMonitor:
    """Class for monitoring news and social media for trading-relevant sentiment"""
    
    def __init__(self, cache_dir=None):
        """
        Initialize the monitor
        
        Parameters:
          - cache_dir: Directory to cache results (None for no caching)
        """
        self.cache_dir = cache_dir or os.path.join(SystemConfig.CACHE_DIR, 'news_social')
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize APIs if keys are available
        if NEWS_API_KEY and NewsApiClient is not None:
            try:
                self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
                logger.info("News API initialized")
            except Exception as e:
                logger.error(f"Error initializing News API: {e}")
                self.news_api = None
        else:
            logger.warning("News API not available - missing API key or NewsApiClient module")
            self.news_api = None
        
        # Initialize Reddit if credentials are available
        if all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]) and praw is not None:
            try:
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                logger.info("Reddit API initialized")
            except Exception as e:
                logger.error(f"Error initializing Reddit API: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit API not available - missing credentials or praw module")
            self.reddit = None
        
        # Initialize StockTwits API (no key required)
        self.stocktwits_api_url = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
        
        # Initialize rate limiter counters
        self.api_calls = {
            'newsapi': {'last_reset': time.time(), 'calls': 0},
            'reddit': {'last_reset': time.time(), 'calls': 0},
            'stocktwits': {'last_reset': time.time(), 'calls': 0},
            'rss': {'last_reset': time.time(), 'calls': 0}
        }
        
        # Rate limits
        self.rate_limits = {
            'newsapi': {'window': 86400, 'max_calls': 100},  # 100 calls per day
            'reddit': {'window': 60, 'max_calls': 60},       # 60 calls per minute
            'stocktwits': {'window': 60, 'max_calls': 30},   # 30 calls per minute
            'rss': {'window': 60, 'max_calls': 10}           # 10 calls per minute
        }
        
        logger.info("NewsSocialMonitor initialized")
    
    def _check_rate_limit(self, api_name):
        """
        Check if we're within rate limits
        
        Parameters:
          - api_name: Name of the API to check
          
        Returns:
          - bool: True if we can proceed, False if rate limited
        """
        if api_name not in self.api_calls or api_name not in self.rate_limits:
            return True
        
        # Get current counters
        api_data = self.api_calls[api_name]
        limits = self.rate_limits[api_name]
        
        # Check if we need to reset the counter
        now = time.time()
        if now - api_data['last_reset'] > limits['window']:
            api_data['last_reset'] = now
            api_data['calls'] = 0
        
        # Check if we're within limits
        if api_data['calls'] >= limits['max_calls']:
            logger.warning(f"Rate limit reached for {api_name}")
            return False
        
        # Increment counter
        api_data['calls'] += 1
        return True
    
    def _cache_key(self, function_name, *args, **kwargs):
        """
        Generate a cache key for the given function and arguments
        
        Parameters:
          - function_name: Name of the function
          - args: Function arguments
          - kwargs: Function keyword arguments
          
        Returns:
          - str: Cache key
        """
        # Convert args and kwargs to a string representation
        args_str = ','.join([str(arg) for arg in args])
        kwargs_str = ','.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        
        # Combine for final key
        return f"{function_name}_{args_str}_{kwargs_str}"
    
    def _get_cached_result(self, key, max_age_seconds=3600):
        """
        Get a cached result if available and not expired
        
        Parameters:
          - key: Cache key
          - max_age_seconds: Maximum age in seconds for cached result
          
        Returns:
          - object or None: Cached result or None if not available
        """
        if not self.cache_dir:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pickle")
        
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            
            if file_age < max_age_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading cache file: {e}")
        
        return None
    
    def _save_cached_result(self, key, result):
        """
        Save a result to cache
        
        Parameters:
          - key: Cache key
          - result: Result to cache
        """
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pickle")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error saving cache file: {e}")
    
    def fetch_news_api(self, query, days_back=3, max_results=10, cache_age=3600):
        """
        Fetch news from News API
        
        Parameters:
          - query: Search query
          - days_back: Number of days to look back
          - max_results: Maximum number of results to return
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - list: News articles
        """
        # Check cache
        cache_key = self._cache_key('news_api', query, days_back, max_results)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached News API results for '{query}'")
            return cached_result
        
        # Check rate limits
        if not self._check_rate_limit('newsapi'):
            logger.warning(f"Rate limit exceeded for News API")
            return []
        
        # Check if API is available
        if self.news_api is None:
            logger.warning("News API not available")
            return []
        
        try:
            # Calculate date range
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Query News API
            response = self.news_api.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page=1,
                page_size=max_results
            )
            
            # Extract articles
            articles = response.get('articles', [])
            
            # Extract relevant fields
            results = []
            for article in articles:
                results.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', '')
                })
            
            # Cache results
            self._save_cached_result(cache_key, results)
            
            logger.info(f"Fetched {len(results)} articles from News API for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
            return []
    
    def fetch_reddit_data(self, query, subreddit="stocks", limit=20, cache_age=3600):
        """
        Fetch posts from Reddit matching the query
        
        Parameters:
          - query: Search query
          - subreddit: Subreddit to search
          - limit: Maximum number of results
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - list: Reddit posts
        """
        # Check cache
        cache_key = self._cache_key('reddit', query, subreddit, limit)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached Reddit results for '{query}' in r/{subreddit}")
            return cached_result
        
        # Check rate limits
        if not self._check_rate_limit('reddit'):
            logger.warning(f"Rate limit exceeded for Reddit API")
            return []
        
        # Check if API is available
        if self.reddit is None:
            logger.warning("Reddit API not available")
            return []
        
        try:
            # Choose subreddit(s)
            if subreddit == "all_finance":
                # Multiple finance-related subreddits
                subreddits = ["stocks", "investing", "wallstreetbets", "finance", "StockMarket"]
                results = []
                
                # Search in each subreddit
                for sub in subreddits:
                    posts = self.reddit.subreddit(sub).search(query, limit=int(limit/len(subreddits)))
                    
                    for post in posts:
                        results.append({
                            'title': post.title,
                            'text': post.selftext if hasattr(post, 'selftext') else '',
                            'score': post.score,
                            'comments': post.num_comments,
                            'url': f"https://www.reddit.com{post.permalink}",
                            'subreddit': sub,
                            'created_utc': post.created_utc
                        })
            else:
                # Single subreddit
                posts = self.reddit.subreddit(subreddit).search(query, limit=limit)
                
                results = []
                for post in posts:
                    results.append({
                        'title': post.title,
                        'text': post.selftext if hasattr(post, 'selftext') else '',
                        'score': post.score,
                        'comments': post.num_comments,
                        'url': f"https://www.reddit.com{post.permalink}",
                        'subreddit': subreddit,
                        'created_utc': post.created_utc
                    })
            
            # Cache results
            self._save_cached_result(cache_key, results)
            
            logger.info(f"Fetched {len(results)} posts from Reddit for '{query}' in r/{subreddit}")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching from Reddit: {e}")
            return []
    
    def fetch_stocktwits_data(self, symbol, limit=30, cache_age=3600):
        """
        Fetch messages from StockTwits for a symbol
        
        Parameters:
          - symbol: Stock symbol
          - limit: Maximum number of messages to return
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - list: StockTwits messages
        """
        # Check cache
        cache_key = self._cache_key('stocktwits', symbol, limit)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached StockTwits results for '{symbol}'")
            return cached_result
        
        # Check rate limits
        if not self._check_rate_limit('stocktwits'):
            logger.warning(f"Rate limit exceeded for StockTwits API")
            return []
        
        try:
            # Clean symbol (remove any special characters)
            clean_symbol = re.sub(r'[^\w\.]', '', symbol.upper())
            
            # Query StockTwits API
            url = self.stocktwits_api_url.format(clean_symbol)
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.warning(f"StockTwits API returned status code {response.status_code}")
                return []
            
            data = response.json()
            
            # Extract messages
            messages = data.get("messages", [])
            
            # Extract relevant fields
            results = []
            for msg in messages[:limit]:
                results.append({
                    'body': msg.get('body', ''),
                    'user': msg.get('user', {}).get('username', ''),
                    'created_at': msg.get('created_at', ''),
                    'sentiment': msg.get('entities', {}).get('sentiment', {}).get('basic', 'neutral')
                })
            
            # Cache results
            self._save_cached_result(cache_key, results)
            
            logger.info(f"Fetched {len(results)} messages from StockTwits for '{symbol}'")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching from StockTwits: {e}")
            return []
    
    def fetch_rss_data(self, query, max_results=15, cache_age=3600):
        """
        Fetch news from RSS feeds
        
        Parameters:
          - query: Search query
          - max_results: Maximum number of results to return
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - list: News articles
        """
        # Check cache
        cache_key = self._cache_key('rss', query, max_results)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached RSS results for '{query}'")
            return cached_result
        
        # Check rate limits
        if not self._check_rate_limit('rss'):
            logger.warning(f"Rate limit exceeded for RSS feeds")
            return []
        
        try:
            # List of RSS feeds to query
            feeds = [
                f"https://news.google.com/rss/search?q={query}",
                f"https://finance.yahoo.com/rss/headline?s={query}",
                f"https://seekingalpha.com/api/sa/combined/{query}.xml"
            ]
            
            results = []
            
            # Query each feed
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    # Extract entries
                    for entry in feed.entries[:int(max_results/len(feeds))]:
                        results.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': feed.feed.get('title', feed_url)
                        })
                except Exception as e:
                    logger.warning(f"Error parsing feed {feed_url}: {e}")
            
            # Cache results
            self._save_cached_result(cache_key, results)
            
            logger.info(f"Fetched {len(results)} articles from RSS feeds for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching from RSS feeds: {e}")
            return []
    
    def get_news_sentiment(self, query, days_back=3, cache_age=3600):
        """
        Get sentiment from news articles for a query
        
        Parameters:
          - query: Search query
          - days_back: Number of days to look back
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - dict: Sentiment results
        """
        # Check cache
        cache_key = self._cache_key('news_sentiment', query, days_back)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached news sentiment for '{query}'")
            return cached_result
        
        try:
            # Fetch news from multiple sources
            news_api_articles = self.fetch_news_api(query, days_back=days_back)
            rss_articles = self.fetch_rss_data(query)
            
            # Combine articles
            articles = news_api_articles + rss_articles
            
            if not articles:
                logger.warning(f"No news articles found for '{query}'")
                return {"sentiment": "NEUTRAL", "score": 0, "articles_analyzed": 0}
            
            # Analyze sentiment for each article
            sentiments = []
            
            for article in articles:
                # Combine title and description/summary for better context
                text = article.get('title', '')
                if 'description' in article and article['description']:
                    text += " " + article['description']
                elif 'summary' in article and article['summary']:
                    text += " " + article['summary']
                
                # Get sentiment
                sentiment = aggregate_sentiments(text)
                sentiments.append(sentiment.get('score', 0))
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Determine overall sentiment label
            if avg_sentiment > 0.2:
                sentiment_label = "POSITIVE"
            elif avg_sentiment < -0.2:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            result = {
                "sentiment": sentiment_label,
                "score": avg_sentiment,
                "articles_analyzed": len(articles)
            }
            
            # Cache result
            self._save_cached_result(cache_key, result)
            
            logger.info(f"News sentiment for '{query}': {sentiment_label} ({avg_sentiment:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {"sentiment": "NEUTRAL", "score": 0, "articles_analyzed": 0}
    
    def get_social_sentiment(self, query, symbol=None, cache_age=3600):
        """
        Get sentiment from social media for a query
        
        Parameters:
          - query: Search query
          - symbol: Stock symbol (optional)
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - dict: Sentiment results
        """
        # Check cache
        cache_key = self._cache_key('social_sentiment', query, symbol)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached social sentiment for '{query}'")
            return cached_result
        
        try:
            # Fetch data from social platforms
            reddit_posts = self.fetch_reddit_data(query, subreddit="all_finance")
            
            # Fetch StockTwits data if symbol is provided
            stocktwits_messages = []
            if symbol:
                stocktwits_messages = self.fetch_stocktwits_data(symbol)
            
            # Combine texts for analysis
            texts = []
            
            # Add Reddit posts
            for post in reddit_posts:
                text = post.get('title', '')
                if post.get('text'):
                    text += " " + post['text']
                texts.append(text)
            
            # Add StockTwits messages
            for msg in stocktwits_messages:
                texts.append(msg.get('body', ''))
            
            if not texts:
                logger.warning(f"No social media content found for '{query}'")
                return {"sentiment": "NEUTRAL", "score": 0, "items_analyzed": 0}
            
            # Analyze sentiment for each text
            sentiments = []
            
            for text in texts:
                sentiment = aggregate_sentiments(text)
                sentiments.append(sentiment.get('score', 0))
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Determine overall sentiment label
            if avg_sentiment > 0.2:
                sentiment_label = "POSITIVE"
            elif avg_sentiment < -0.2:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            # Include StockTwits sentiment analysis if available
            stocktwits_sentiment = 0
            stocktwits_count = 0
            
            for msg in stocktwits_messages:
                sentiment = msg.get('sentiment')
                if sentiment == 'bullish':
                    stocktwits_sentiment += 1
                    stocktwits_count += 1
                elif sentiment == 'bearish':
                    stocktwits_sentiment -= 1
                    stocktwits_count += 1
            
            stocktwits_score = stocktwits_sentiment / stocktwits_count if stocktwits_count > 0 else 0
            
            # Combine NLP and StockTwits sentiment (if available)
            if stocktwits_count > 0:
                combined_score = (avg_sentiment + stocktwits_score) / 2
            else:
                combined_score = avg_sentiment
            
            result = {
                "sentiment": sentiment_label,
                "score": combined_score,
                "items_analyzed": len(texts),
                "reddit_posts": len(reddit_posts),
                "stocktwits_messages": len(stocktwits_messages)
            }
            
            # Cache result
            self._save_cached_result(cache_key, result)
            
            logger.info(f"Social sentiment for '{query}': {sentiment_label} ({combined_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
            return {"sentiment": "NEUTRAL", "score": 0, "items_analyzed": 0}
    
    def get_combined_sentiment(self, query, symbol=None, days_back=3, cache_age=3600):
        """
        Get combined sentiment from news and social media
        
        Parameters:
          - query: Search query
          - symbol: Stock symbol (optional)
          - days_back: Number of days to look back for news
          - cache_age: Maximum age for cached results in seconds
          
        Returns:
          - dict: Combined sentiment results
        """
        # Check cache
        cache_key = self._cache_key('combined_sentiment', query, symbol, days_back)
        cached_result = self._get_cached_result(cache_key, cache_age)
        
        if cached_result is not None:
            logger.debug(f"Using cached combined sentiment for '{query}'")
            return cached_result
        
        try:
            # Get sentiment from different sources
            news_sentiment = self.get_news_sentiment(query, days_back)
            social_sentiment = self.get_social_sentiment(query, symbol)
            
            # Combine with weighting (news: 60%, social: 40%)
            news_weight = 0.6
            social_weight = 0.4
            
            combined_score = (
                news_sentiment.get('score', 0) * news_weight +
                social_sentiment.get('score', 0) * social_weight
            )
            
            # Determine overall sentiment label
            if combined_score > 0.2:
                sentiment_label = "POSITIVE"
            elif combined_score < -0.2:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            result = {
                "sentiment": sentiment_label,
                "score": combined_score,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment
            }
            
            # Cache result
            self._save_cached_result(cache_key, result)
            
            logger.info(f"Combined sentiment for '{query}': {sentiment_label} ({combined_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error getting combined sentiment: {e}")
            return {"sentiment": "NEUTRAL", "score": 0}
    
    def get_news_summary(self, query, days_back=3, max_articles=5):
        """
        Get a summary of the latest news for a query
        
        Parameters:
          - query: Search query
          - days_back: Number of days to look back
          - max_articles: Maximum number of articles to include
          
        Returns:
          - str: News summary
        """
        try:
            # Fetch news from multiple sources
            news_api_articles = self.fetch_news_api(query, days_back=days_back)
            rss_articles = self.fetch_rss_data(query)
            
            # Combine and sort by date (newest first)
            articles = news_api_articles + rss_articles
            
            # Sort articles by date if available
            def get_date(article):
                # Try different date fields
                for field in ['published_at', 'published', 'date']:
                    if field in article and article[field]:
                        try:
                            return pd.to_datetime(article[field])
                        except:
                            pass
                return pd.Timestamp.now()
            
            articles.sort(key=get_date, reverse=True)
            
            # Limit to max_articles
            articles = articles[:max_articles]
            
            if not articles:
                return f"No news found for '{query}' in the past {days_back} days."
            
            # Generate summary
            summary = f"Latest news for {query}:\n\n"
            
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source = article.get('source', 'Unknown source')
                date = get_date(article).strftime('%Y-%m-%d')
                
                summary += f"{i}. {title} ({source}, {date})\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting news summary: {e}")
            return f"Error retrieving news for '{query}'"

# Legacy function for backward compatibility
def get_combined_sentiment(query, symbol=None, days_back=3):
    """
    Legacy function to maintain compatibility with original code
    
    Parameters:
      - query: Search query
      - symbol: Stock symbol (optional)
      - days_back: Number of days to look back
      
    Returns:
      - dict: Sentiment results
    """
    monitor = NewsSocialMonitor()
    return monitor.get_combined_sentiment(query, symbol, days_back)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    monitor = NewsSocialMonitor()
    
    # Get combined sentiment for a stock
    symbol = "AAPL"
    query = f"{symbol} stock"
    
    sentiment = monitor.get_combined_sentiment(query, symbol)
    print(f"Combined sentiment for {symbol}: {sentiment['sentiment']} ({sentiment['score']:.2f})")
    
    # Get news summary
    summary = monitor.get_news_summary(query)
    print("\nNews Summary:")
    print(summary)