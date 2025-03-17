# main.py
import pandas as pd
from modules.data_acquisition import fetch_price_data
from modules.ml_predictor import train_ensemble_model, predict_with_model
from modules.logger import log_trade

# Fetch data
data = fetch_price_data('AAPL', '2022-01-01', '2023-01-01')
data.to_csv('data/historical_prices.csv')

# Prepare data
data['target'] = data['Close'].shift(-1)
data.dropna(inplace=True)
train_data = data.iloc[:-30]
test_data = data.iloc[-30:].drop(columns='target')

# Train and predict with AutoGluon
predictor = train_ensemble_model(train_data)
predictions = predict_with_model(predictor, test_data)

print("Predictions:", predictions)

# Log predictions
from modules.logger import log_trade
log_trade(f"AutoGluon predictions generated successfully: {predictions.mean()}")
