import unittest
import numpy as np
from modules.ml_predictor import lstm_predict, prophet_predict, ensemble_predict

class TestMLPredictor(unittest.TestCase):
    def test_lstm_predict(self):
        X = np.random.rand(10, 5)
        preds = lstm_predict(X)
        self.assertEqual(len(preds), 10)
    
    def test_prophet_predict(self):
        X = np.random.rand(10, 5)
        preds = prophet_predict(X)
        self.assertEqual(len(preds), 10)
    
    def test_ensemble_predict(self):
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(20, 10)
        preds = ensemble_predict(X_train, y_train, X_test)
        self.assertEqual(len(preds), 20)

if __name__ == '__main__':
    unittest.main()
