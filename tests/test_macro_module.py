import unittest
from modules.macro_module import get_gdp_indicator

class TestMacroModule(unittest.TestCase):
    def test_get_gdp_indicator(self):
        # Ensure that the function returns either "BULLISH" or "BEARISH"
        result = get_gdp_indicator()
        self.assertIn(result, ["BULLISH", "BEARISH"])

if __name__ == '__main__':
    unittest.main()
