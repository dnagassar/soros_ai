import unittest
from modules.logger import log_trade

class TestLogger(unittest.TestCase):
    def test_log_trade(self):
        test_message = "Test log entry for unit test"
        log_trade(test_message)
        with open("trading.log", "r") as f:
            contents = f.read()
        self.assertIn(test_message, contents)

if __name__ == '__main__':
    unittest.main()
