import unittest
from modules.news_social_monitor import get_aggregated_sentiment

class TestNewsSocialMonitor(unittest.TestCase):
    def test_get_aggregated_sentiment(self):
        result = get_aggregated_sentiment("AAPL")
        self.assertIn("sentiment", result)
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], (float, int))

if __name__ == '__main__':
    unittest.main()
