import unittest
from modules.earnings_module import get_upcoming_earnings

class TestEarningsModule(unittest.TestCase):
    def test_get_upcoming_earnings(self):
        result = get_upcoming_earnings("AAPL")
        self.assertIn(result, ["EVENT_PENDING", "NO_EVENT"])

if __name__ == '__main__':
    unittest.main()
