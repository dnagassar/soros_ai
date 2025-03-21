# modules/sentiment_analysis.py
"""
Sentiment analysis module for analyzing text data from news and social media
"""
import openai
import nltk
import re
import os
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
from config import OPENAI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Cache directory for sentiment results
CACHE_DIR = 'cache/sentiment'
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_text(text):
    """Clean and normalize text for sentiment analysis"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?:;\'\"()-]', '', text)
    
    return text

def get_cached_sentiment(text_hash):
    """Check if we have a cached sentiment result for the text"""
    cache_file = os.path.join(CACHE_DIR, f"{text_hash}.pickle")
    
    if os.path.exists(cache_file):
        try:
            # Check if cache is still valid (less than 24 hours old)
            if datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached sentiment: {e}")
    
    return None

def save_cached_sentiment(text_hash, result):
    """Save sentiment result to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{text_hash}.pickle")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Error saving sentiment to cache: {e}")

def analyze_with_vader(text):
    """Analyze sentiment using VADER"""
    try:
        sentiment = vader.polarity_scores(text)
        
        # Convert compound score to -1 to 1 range
        score = sentiment['compound']
        
        # Determine sentiment label
        if score >= 0.05:
            label = "POSITIVE"
        elif score <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "score": score,
            "label": label,
            "details": {
                "positive": sentiment["pos"],
                "negative": sentiment["neg"],
                "neutral": sentiment["neu"]
            },
            "method": "vader"
        }
    except Exception as e:
        logger.error(f"Error in VADER sentiment analysis: {e}")
        return {
            "score": 0,
            "label": "NEUTRAL",
            "details": {"error": str(e)},
            "method": "vader"
        }

def analyze_with_openai(text, symbol=None):
    """Analyze sentiment using OpenAI API"""
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured, falling back to VADER")
        return analyze_with_vader(text)
    
    try:
        # Limit text length for API call
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        prompt = f"""
        Analyze the sentiment of the following text about {symbol if symbol else 'a financial asset'}.
        Return a JSON object with:
        1. "score": a score from -1 (extremely negative) to 1 (extremely positive)
        2. "label": "POSITIVE", "NEGATIVE", or "NEUTRAL"
        3. "summary": a brief 1-2 sentence summary of the key sentiment drivers
        
        Text to analyze:
        "{text}"
        
        JSON response:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a financial sentiment analysis expert. Analyze the given text and return JSON with sentiment analysis results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for consistent results
            max_tokens=150,   # Keep response concise
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all expected fields are present
        result.setdefault("score", 0)
        result.setdefault("label", "NEUTRAL")
        result.setdefault("summary", "No summary provided")
        result["method"] = "openai"
        
        return result
    
    except Exception as e:
        logger.error(f"Error in OpenAI sentiment analysis: {e}, falling back to VADER")
        return analyze_with_vader(text)

def analyze_sentiment(text, method="hybrid", symbol=None):
    """
    Analyze sentiment of text using specified method
    
    Parameters:
      - text: Text to analyze
      - method: Sentiment analysis method ('vader', 'openai', or 'hybrid')
      - symbol: Optional symbol for context
      
    Returns:
      - dict: Sentiment analysis result
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Generate a simple hash for caching
    text_hash = str(hash(cleaned_text + method + (symbol or "")))
    
    # Check cache
    cached_result = get_cached_sentiment(text_hash)
    if cached_result:
        return cached_result
    
    # If no text, return neutral
    if not cleaned_text:
        result = {
            "score": 0,
            "label": "NEUTRAL",
            "details": {"error": "No text provided"},
            "method": method
        }
        save_cached_sentiment(text_hash, result)
        return result
    
    # Analyze based on method
    if method == "vader":
        result = analyze_with_vader(cleaned_text)
    elif method == "openai":
        result = analyze_with_openai(cleaned_text, symbol)
    else:  # hybrid approach
        try:
            # First try with OpenAI if API key is available
            if OPENAI_API_KEY:
                result = analyze_with_openai(cleaned_text, symbol)
            else:
                # Fall back to VADER if no API key
                result = analyze_with_vader(cleaned_text)
        except Exception:
            # Fall back to VADER on any error
            result = analyze_with_vader(cleaned_text)
    
    # Add timestamp
    result["timestamp"] = datetime.now().isoformat()
    
    # Cache result
    save_cached_sentiment(text_hash, result)
    
    return result

def aggregate_sentiments(texts, weights=None, method="hybrid", symbol=None):
    """
    Aggregate sentiments from multiple text sources
    
    Parameters:
      - texts: List of texts or single text
      - weights: Optional weights for each text
      - method: Sentiment analysis method
      - symbol: Optional symbol for context
      
    Returns:
      - dict: Aggregated sentiment result
    """
    # Handle single text case
    if isinstance(texts, str):
        return analyze_sentiment(texts, method, symbol)
    
    # Empty list case
    if not texts:
        return {
            "score": 0,
            "label": "NEUTRAL",
            "details": {"error": "No texts provided"},
            "method": method
        }
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = [1.0 / len(texts)] * len(texts)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(texts)] * len(texts)
    
    # Analyze each text
    results = []
    for text in texts:
        results.append(analyze_sentiment(text, method, symbol))
    
    # Calculate weighted average score
    weighted_score = sum(r["score"] * w for r, w in zip(results, weights))
    
    # Determine label
    if weighted_score >= 0.05:
        label = "POSITIVE"
    elif weighted_score <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    # Collect summaries if available
    summaries = [r.get("summary") for r in results if r.get("summary")]
    
    return {
        "score": weighted_score,
        "label": label,
        "details": {
            "individual_results": results,
            "weights": weights,
            "summaries": summaries[:3]  # Limit to top 3 summaries
        },
        "method": method,
        "timestamp": datetime.now().isoformat()
    }

def get_market_sentiment_score(symbol=None, lookback_days=1):
    """
    Get overall market sentiment score
    
    Parameters:
      - symbol: Optional specific symbol (None for market)
      - lookback_days: Days to look back for sentiment
      
    Returns:
      - float: Sentiment score (-1 to 1)
    """
    # In a real implementation, this would aggregate from multiple sources
    # For now, return a mildly positive sentiment as a placeholder
    return 0.2

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test individual text analysis
    test_text = "The company reported strong earnings, beating analyst expectations. Revenue grew by 15% year-over-year, but there are concerns about rising costs."
    
    result = analyze_sentiment(test_text, method="vader", symbol="AAPL")
    print(f"VADER result: {json.dumps(result, indent=2)}")
    
    if OPENAI_API_KEY:
        result = analyze_sentiment(test_text, method="openai", symbol="AAPL")
        print(f"OpenAI result: {json.dumps(result, indent=2)}")
    
    # Test aggregation
    texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "Analysts have raised concerns about the company's high debt levels.",
        "New product launch planned for next quarter, expected to drive growth."
    ]
    
    weights = [0.5, 0.3, 0.2]
    
    result = aggregate_sentiments(texts, weights, method="vader", symbol="AAPL")
    print(f"Aggregated result: {json.dumps(result, indent=2)}")