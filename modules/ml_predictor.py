# ml_predictor.py
import pandas as pd
from autogluon.tabular import TabularPredictor

def train_ensemble_model(train_data: pd.DataFrame, label_column='target', eval_metric='root_mean_squared_error'):
    predictor = TabularPredictor(label='target', eval_metric='rmse').fit(
        train_data=train_data, 
        presets='best_quality',
        time_limit=300  # 5-minute quick run; increase for accuracy
    )
    return predictor

def predict_with_model(predictor, test_data):
    predictions = predictor.predict(test_data)
    return predictions

if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv('../data/historical_prices.csv')
    df['target'] = df['Close'].shift(-1)  # Predict next day's price
    df.dropna(inplace=True)

    train_data = df.iloc[:-20]
    test_data = df.iloc[-20:].drop(columns=['target'])

    predictor = train_ensemble_model(train_data=df)
    preds = predict_with_model(predictor, test_data)

    print("Ensemble Predictions:\n", predictions)
