# tests/test_enhanced_sentiment_analysis.py
import unittest
import sys
import os
import json
import numpy as np
from datetime import datetime

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sentiment analysis module
from modules.sentiment_analysis import (
    clean_text,
    get_cached_sentiment,
    save_cached_sentiment,
    analyze_with_vader,
    analyze_with_finbert,
    analyze_sentiment,
    aggregate_sentiments,
    get_market_sentiment_score,
    initialize_finbert
)

class TestEnhancedSentimentAnalysis(unittest.TestCase):
    """
    Test suite for the enhanced sentiment analysis module with both VADER and FinBERT
    """
    
    def setUp(self):
        """Set up test environment"""
        # Example texts for testing
        self.positive_text = "The company reported excellent earnings, significantly beating expectations."
        self.negative_text = "The stock crashed after the company missed earnings by a wide margin."
        self.neutral_text = "The company released its quarterly earnings report today."
        
        # Batch of texts for aggregation testing
        self.text_batch = [
            "The company reported strong earnings, beating analyst expectations.",
            "However, analysts have raised concerns about rising debt levels.",
            "A new product launch is expected to drive future growth."
        ]
        
        # Create test directory for cache testing
        self.test_cache_dir = "test_cache"
        os.makedirs(self.test_cache_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test cache files
        for f in os.listdir(self.test_cache_dir):
            os.remove(os.path.join(self.test_cache_dir, f))
        os.rmdir(self.test_cache_dir)
    
    def test_clean_text(self):
        """Test text cleaning function"""
        # Test with URLs
        text_with_url = "Check this link: https://example.com and read more."
        cleaned = clean_text(text_with_url)
        self.assertNotIn("https://", cleaned)
        
        # Test with HTML
        text_with_html = "<p>This is a <b>paragraph</b> with HTML.</p>"
        cleaned = clean_text(text_with_html)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        
        # Test with extra whitespace
        text_with_whitespace = "  Too   many    spaces   "
        cleaned = clean_text(text_with_whitespace)
        self.assertEqual(cleaned, "Too many spaces")
        
        # Test with special characters
        text_with_special = "Symbol: @#$%^&* but keep punctuation: .,!?()-"
        cleaned = clean_text(text_with_special)
        self.assertNotIn("@#$%^&*", cleaned)
        self.assertIn(".,!?()-", cleaned)
        
        # Test with empty input
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
    
    def test_cache_functions(self):
        """Test caching functions"""
        test_hash = "test_hash_123"
        test_method = "vader"
        test_result = {"score": 0.5, "label": "POSITIVE"}
        
        # Test saving to cache
        save_cached_sentiment(test_hash, test_method, test_result)
        
        # Test retrieving from cache
        cached_result = get_cached_sentiment(test_hash, test_method)
        self.assertEqual(cached_result, test_result)
        
        # Test with non-existent cache
        self.assertIsNone(get_cached_sentiment("nonexistent_hash", test_method))
    
    def test_vader_analysis(self):
        """Test VADER sentiment analysis"""
        # Test positive sentiment
        positive_result = analyze_with_vader(self.positive_text)
        self.assertGreater(positive_result["score"], 0)
        self.assertEqual(positive_result["label"], "POSITIVE")
        self.assertEqual(positive_result["method"], "vader")
        
        # Test negative sentiment
        negative_result = analyze_with_vader(self.negative_text)
        self.assertLess(negative_result["score"], 0)
        self.assertEqual(negative_result["label"], "NEGATIVE")
        
        # Test neutral sentiment
        neutral_result = analyze_with_vader(self.neutral_text)
        self.assertIn(neutral_result["label"], ["NEUTRAL", "POSITIVE", "NEGATIVE"])
        
        # Test error handling
        try:
            result = analyze_with_vader(None)
            self.assertEqual(result["label"], "NEUTRAL")
        except Exception as e:
            self.fail(f"analyze_with_vader raised an exception with None input: {e}")
    
    def test_finbert_analysis(self):
        """Test FinBERT sentiment analysis"""
        try:
            # Initialize FinBERT
            if not initialize_finbert():
                self.skipTest("FinBERT could not be initialized, skipping tests")
            
            # Test positive sentiment
            positive_result = analyze_with_finbert(self.positive_text)
            self.assertEqual(positive_result["method"], "finbert")
            
            # Test negative sentiment
            negative_result = analyze_with_finbert(self.negative_text)
            
            # Test neutral sentiment
            neutral_result = analyze_with_finbert(self.neutral_text)
            
            # Check that we have score and label in all results
            for result in [positive_result, negative_result, neutral_result]:
                self.assertIn("score", result)
                self.assertIn("label", result)
                self.assertIn(result["label"], ["POSITIVE", "NEGATIVE", "NEUTRAL"])
            
            # Test error handling
            try:
                result = analyze_with_finbert(None)
                self.assertEqual(result["label"], "NEUTRAL")
            except Exception as e:
                self.fail(f"analyze_with_finbert raised an exception with None input: {e}")
                
        except ImportError:
            self.skipTest("FinBERT dependencies not installed, skipping tests")
    
    def test_analyze_sentiment(self):
        """Test the main sentiment analysis function with both methods"""
        # Test with VADER
        vader_result = analyze_sentiment(self.positive_text, method="vader")
        self.assertEqual(vader_result["method"], "vader")
        
        # Test with FinBERT (if available)
        try:
            finbert_result = analyze_sentiment(self.positive_text, method="finbert")
            self.assertEqual(finbert_result["method"], "finbert")
        except:
            # Skip if FinBERT is not available
            print("FinBERT not available, skipping that part of the test")
        
        # Test with empty text
        empty_result = analyze_sentiment("")
        self.assertEqual(empty_result["label"], "NEUTRAL")
        self.assertEqual(empty_result["score"], 0)
        
        # Test with symbol context
        symbol_result = analyze_sentiment(self.positive_text, method="vader", symbol="AAPL")
        self.assertIn("score", symbol_result)
        self.assertIn("label", symbol_result)
    
    def test_aggregate_sentiments(self):
        """Test sentiment aggregation function"""
        # Test with a single text
        single_result = aggregate_sentiments(self.positive_text)
        self.assertIsInstance(single_result, dict)
        self.assertIn("score", single_result)
        
        # Test with a batch of texts
        batch_result = aggregate_sentiments(self.text_batch)
        self.assertIsInstance(batch_result, dict)
        self.assertIn("score", batch_result)
        
        # Test with weights
        weights = [0.5, 0.3, 0.2]
        weighted_result = aggregate_sentiments(self.text_batch, weights)
        self.assertIsInstance(weighted_result, dict)
        
        # Test with empty list
        empty_result = aggregate_sentiments([])
        self.assertEqual(empty_result["label"], "NEUTRAL")
        self.assertEqual(empty_result["score"], 0)
        
        # Test with different methods
        vader_result = aggregate_sentiments(self.text_batch, method="vader")
        self.assertEqual(vader_result["method"], "vader")
        
        try:
            finbert_result = aggregate_sentiments(self.text_batch, method="finbert")
            self.assertEqual(finbert_result["method"], "finbert")
        except:
            print("FinBERT not available, skipping that part of the test")
    
    def test_get_market_sentiment_score(self):
        """Test market sentiment score function"""
        # Test general market sentiment
        market_score = get_market_sentiment_score()
        self.assertIsInstance(market_score, float)
        self.assertTrue(-1 <= market_score <= 1)
        
        # Test symbol-specific sentiment
        symbol_score = get_market_sentiment_score(symbol="AAPL")
        self.assertIsInstance(symbol_score, float)
        self.assertTrue(-1 <= symbol_score <= 1)
        
        # Test with different methods
        vader_score = get_market_sentiment_score(method="vader")
        self.assertIsInstance(vader_score, float)
        
        try:
            finbert_score = get_market_sentiment_score(method="finbert")
            self.assertIsInstance(finbert_score, float)
        except:
            print("FinBERT not available, skipping that part of the test")
    
    def test_performance(self):
        """Test performance of sentiment analysis"""
        # Generate a larger batch of texts
        large_batch = [
            f"Sample text {i} for performance testing." for i in range(20)
        ]
        
        # Test VADER performance
        start_time = datetime.now()
        vader_result = aggregate_sentiments(large_batch, method="vader")
        vader_time = (datetime.now() - start_time).total_seconds()
        
        # Print performance results
        print(f"\nVADER processed 20 texts in {vader_time:.2f} seconds")
        
        # Test FinBERT performance if available
        try:
            start_time = datetime.now()
            finbert_result = aggregate_sentiments(large_batch, method="finbert")
            finbert_time = (datetime.now() - start_time).total_seconds()
            print(f"FinBERT processed 20 texts in {finbert_time:.2f} seconds")
            print(f"FinBERT is {finbert_time/vader_time:.1f}x slower than VADER")
        except:
            print("FinBERT not available, skipping performance comparison")

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    # Check NLTK
    try:
        import nltk
    except ImportError:
        missing.append("nltk")
    
    # Check Transformers
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    # Check PyTorch
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    return missing

def run_tests():
    """Run the tests and catch any import errors"""
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped.")
    
    # Run the tests
    unittest.main()

if __name__ == "__main__":
    run_tests()