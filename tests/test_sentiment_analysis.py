import unittest
from modules.sentiment_analysis import analyze_sentiment_hf, analyze_sentiment_vader, aggregate_sentiments

class TestSentimentAnalysis(unittest.TestCase):
    def test_analyze_sentiment_hf(self):
        result = analyze_sentiment_hf("The market is bullish")
        self.assertIn('label', result)
        self.assertIn('score', result)

    def test_analyze_sentiment_vader(self):
        result = analyze_sentiment_vader("The market is bearish")
        self.assertIn('compound', result)

    def test_aggregate_sentiments(self):
        result = aggregate_sentiments("The market is bullish")
        self.assertIn('sentiment', result)
        self.assertIn('score', result)

if __name__ == '__main__':
    unittest.main()
