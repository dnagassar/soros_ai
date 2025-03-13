import unittest
import os
from modules.logger import log_trade

class TestLogger(unittest.TestCase):
    def test_log_trade(self):
        log_trade("Test log entry")
        self.assertTrue(os.path.exists("trading.log"))
        with open("trading.log", "r") as f:
            logs = f.read()
        self.assertIn("Test log entry", logs)

if __name__ == '__main__':
    unittest.main()
