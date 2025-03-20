import unittest
import pandas as pd
import numpy as np
from modules.ml_predictor import train_ensemble_model, predict_with_model, ensemble_predict

class TestMLPredictor(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset with 500 samples
        self.df = pd.DataFrame({
            'feature1': np.random.rand(500),
            'feature2': np.random.rand(500),
            'Close': np.random.rand(500) * 100
        })
        self.df['target'] = self.df['Close'].shift(-1)
        self.df.dropna(inplace=True)
        self.train_data = self.df.iloc[:-50]
        self.test_data = self.df.iloc[-50:].drop(columns=['target'])
        self.X_train = self.train_data.drop(columns=['target'])
        self.y_train = self.train_data['target']
        self.X_test = self.test_data

    def test_train_ensemble_model(self):
        predictor = train_ensemble_model(self.train_data, time_limit=60)
        self.assertTrue(isinstance(predictor, object))

    def test_predict_with_model(self):
        predictor = train_ensemble_model(self.train_data, time_limit=60)
        preds = predict_with_model(predictor, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_ensemble_predict(self):
        preds = ensemble_predict(self.X_train, self.y_train, self.X_test, time_limit=60)
        self.assertEqual(len(preds), len(self.X_test))

if __name__ == '__main__':
    unittest.main()
