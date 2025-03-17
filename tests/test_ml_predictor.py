import unittest
import pandas as pd
from modules.ml_predictor import train_ensemble_model, predict_with_model

class TestMLPredictor(unittest.TestCase):
    def test_model_training_and_prediction(self):
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [x * 0.5 for x in range(100)]
        })

        train_data = df.iloc[:80]
        test_data = df.iloc[80:].drop(columns=['target'])

        predictor = train_ensemble_model(train_data=train_data)
        preds = predict_with_model(predictor, test_data)

        self.assertEqual(len(preds), len(test_data))
        self.assertIsNotNone(preds)

if __name__ == '__main__':
    unittest.main()
