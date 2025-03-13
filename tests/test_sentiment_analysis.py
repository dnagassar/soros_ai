import unittest
from modules.sentiment_analysis import analyze_sentiment_hf, analyze_sentiment_vader, aggregate_sentiments

class TestSentimentAnalysis(unittest.TestCase):
    def test_analyze_sentiment_hf(self):
        text = "Market is bullish."
        result = analyze_sentiment_hf(text)
        self.assertIn('label', result)
        self.assertIn('score', result)
    
    def test_analyze_sentiment_vader(self):
        text = "Market is bullish."
        result = analyze_sentiment_vader(text)
        self.assertIn('compound', result)
    
    def test_aggregate_sentiments(self):
        text = "Market is bullish."
        result = aggregate_sentiments(text)
        self.assertIn('sentiment', result)
        self.assertIn('score', result)

if __name__ == '__main__':
    unittest.main()
