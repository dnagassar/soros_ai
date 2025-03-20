import unittest
from modules.asset_selector import fetch_asset_universe, select_top_assets

class TestAssetSelector(unittest.TestCase):
    def test_fetch_asset_universe(self):
        tickers = fetch_asset_universe()
        self.assertIsInstance(tickers, list)
        self.assertGreater(len(tickers), 0)

    def test_select_top_assets(self):
        top_assets = select_top_assets(n=5, use_social=False)
        self.assertIsInstance(top_assets, list)
        self.assertLessEqual(len(top_assets), 5)

if __name__ == '__main__':
    unittest.main()
