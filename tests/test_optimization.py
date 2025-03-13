import unittest
from modules.optimization import optimize_parameters

class TestOptimization(unittest.TestCase):
    def test_optimize_parameters(self):
        # Run a quick optimization (fewer evaluations for testing purposes)
        best_params = optimize_parameters(max_evals=10)
        self.assertIn('lookback_period', best_params)
        self.assertIn('threshold', best_params)

if __name__ == '__main__':
    unittest.main()
