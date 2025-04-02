# modules/sentiment_analysis.py
"""
Enhanced sentiment analysis module for analyzing text data from news and social media
with support for both VADER and FinBERT models
"""
import nltk
import re
import os
import json
import logging
import time
import pickle
import torch
import numpy as np
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import OPENAI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

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

# FinBERT model and tokenizer (lazy-loaded when needed)
finbert_model = None
finbert_tokenizer = None

def initialize_finbert():
    """Initialize FinBERT model and tokenizer"""
    global finbert_model, finbert_tokenizer
    
    try:
        # Load FinBERT model and tokenizer
        model_name = "ProsusAI/finbert"
        finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("FinBERT model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing FinBERT model: {e}")
        return False

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

def get_cached_sentiment(text_hash, method):
    """Check if we have a cached sentiment result for the text"""
    cache_file = os.path.join(CACHE_DIR, f"{text_hash}_{method}.pickle")
    
    if os.path.exists(cache_file):
        try:
            # Check if cache is still valid (less than 24 hours old)
            if datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached sentiment: {e}")
    
    return None

def save_cached_sentiment(text_hash, method, result):
    """Save sentiment result to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{text_hash}_{method}.pickle")
    
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

def analyze_with_finbert(text):
    """Analyze sentiment using FinBERT"""
    global finbert_model, finbert_tokenizer
    
    # Initialize FinBERT if needed
    if finbert_model is None or finbert_tokenizer is None:
        if not initialize_finbert():
            logger.warning("Failed to initialize FinBERT, falling back to VADER")
            return analyze_with_vader(text)
    
    try:
        # Tokenize the text
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            scores = outputs.logits.softmax(dim=1)[0].tolist()
        
        # FinBERT class order: negative, neutral, positive
        label_map = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        predicted_class = scores.index(max(scores))
        label = label_map[predicted_class]
        
        # Convert to a -1 to 1 score 
        # (negative=-1, neutral=0, positive=1, weighted by confidence)
        if predicted_class == 0:  # Negative
            score = -scores[0]
        elif predicted_class == 2:  # Positive
            score = scores[2]
        else:  # Neutral
            score = 0
        
        return {
            "score": score,
            "label": label,
            "details": {
                "negative": scores[0],
                "neutral": scores[1],
                "positive": scores[2]
            },
            "method": "finbert"
        }
    except Exception as e:
        logger.error(f"Error in FinBERT sentiment analysis: {e}")
        return {
            "score": 0,
            "label": "NEUTRAL",
            "details": {"error": str(e)},
            "method": "finbert"
        }

def analyze_sentiment(text, method="vader", symbol=None):
    """
    Analyze sentiment of text using specified method
    
    Parameters:
      - text: Text to analyze
      - method: Sentiment analysis method ('vader', 'finbert')
      - symbol: Optional symbol for context
      
    Returns:
      - dict: Sentiment analysis result
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Generate a hash for caching
    text_hash = str(hash(cleaned_text + (symbol or "")))
    
    # Check cache
    cached_result = get_cached_sentiment(text_hash, method)
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
        save_cached_sentiment(text_hash, method, result)
        return result
    
    # Analyze based on method
    if method.lower() == "finbert":
        result = analyze_with_finbert(cleaned_text)
    else:  # Default to VADER
        result = analyze_with_vader(cleaned_text)
    
    # Add timestamp
    result["timestamp"] = datetime.now().isoformat()
    
    # Cache result
    save_cached_sentiment(text_hash, method, result)
    
    return result

def aggregate_sentiments(texts, weights=None, method="vader", symbol=None):
    """
    Aggregate sentiments from multiple text sources
    
    Parameters:
      - texts: List of texts or single text
      - weights: Optional weights for each text
      - method: Sentiment analysis method ('vader', 'finbert')
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
    
    return {
        "score": weighted_score,
        "label": label,
        "details": {
            "individual_results": results,
            "weights": weights
        },
        "method": method,
        "timestamp": datetime.now().isoformat()
    }

def get_market_sentiment_score(symbol=None, lookback_days=1, method="vader"):
    """
    Get overall market sentiment score
    
    Parameters:
      - symbol: Optional specific symbol (None for market)
      - lookback_days: Days to look back for sentiment
      - method: Sentiment analysis method ('vader', 'finbert')
      
    Returns:
      - float: Sentiment score (-1 to 1)
    """
    # In a real implementation, this would aggregate from multiple sources
    # For now, return a placeholder implementation
    
    # Generate search query based on symbol
    if symbol:
        query = f"{symbol} stock market news"
    else:
        query = "stock market outlook economy"
    
    try:
        # This is a placeholder - in a real system, you would fetch actual news
        sample_texts = [
            f"Latest market updates for {symbol or 'the market'}",
            f"Recent financial news about {symbol or 'global markets'}"
        ]
        
        # Analyze sentiment
        result = aggregate_sentiments(sample_texts, method=method, symbol=symbol)
        return result["score"]
        
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        return 0.0  # Neutral sentiment on error

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Test individual text analysis with both methods
    test_text = "The company reported strong earnings, beating analyst expectations. Revenue grew by 15% year-over-year, but there are concerns about rising costs."
    
    vader_result = analyze_sentiment(test_text, method="vader", symbol="AAPL")
    print(f"VADER result: {json.dumps(vader_result, indent=2)}")
    
    finbert_result = analyze_sentiment(test_text, method="finbert", symbol="AAPL")
    print(f"FinBERT result: {json.dumps(finbert_result, indent=2)}")
    
    # Test aggregation
    texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "Analysts have raised concerns about the company's high debt levels.",
        "New product launch planned for next quarter, expected to drive growth."
    ]
    
    weights = [0.5, 0.3, 0.2]
    
    vader_agg_result = aggregate_sentiments(texts, weights, method="vader", symbol="AAPL")
    print(f"VADER Aggregated result: {json.dumps(vader_agg_result, indent=2)}")
    
    finbert_agg_result = aggregate_sentiments(texts, weights, method="finbert", symbol="AAPL")
    print(f"FinBERT Aggregated result: {json.dumps(finbert_agg_result, indent=2)}")
    
    # Test market sentiment score
    market_score = get_market_sentiment_score(method="vader")
    print(f"Market sentiment score (VADER): {market_score}")
    
    symbol_score = get_market_sentiment_score(symbol="AAPL", method="finbert")
    print(f"AAPL sentiment score (FinBERT): {symbol_score}")