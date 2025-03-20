import unittest
from modules.macro_module import get_gdp_indicator

class TestMacroModule(unittest.TestCase):
    def test_get_gdp_indicator(self):
        result = get_gdp_indicator()
        # Allow NEUTRAL as well
        self.assertIn(result, ["BULLISH", "BEARISH", "NEUTRAL"])

if __name__ == '__main__':
    unittest.main()
