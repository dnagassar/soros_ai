# main.py
import pandas as pd
from modules.data_acquisition import fetch_price_data
from modules.ml_predictor import train_ensemble_model, predict_with_model
from modules.news_social_monitor import get_combined_sentiment
from modules.signal_aggregator import aggregate_signals
from modules.asset_selector import select_top_assets
from modules.logger import log_trade

# 1. Data Acquisition: Fetch historical data for AAPL and save it.
data = fetch_price_data('AAPL', '2022-01-01', '2023-01-01')
data.to_csv('data/historical_prices.csv', index=False)

# 2. Data Preparation: Create target as next day's Close price.
data['target'] = data['Close'].shift(-1)
data.dropna(inplace=True)
train_data = data.iloc[:-30]
test_data = data.iloc[-30:].drop(columns=['target'])

# 3. Train ML Ensemble Model and Generate Predictions.
try:
    predictor = train_ensemble_model(train_data=train_data, time_limit=600)
    predictions = predict_with_model(predictor, test_data)
except Exception as e:
    log_trade(f"Error during ML training/prediction: {e}")
    raise

# 4. Get Combined Social Sentiment for AAPL using Reddit, StockTwits, and RSS feeds.
social_sentiment = get_combined_sentiment("AAPL", symbol="AAPL")
print("Combined Social Sentiment for AAPL:", social_sentiment)

# 5. Aggregate Signals.
news_text = "AAPL is reporting record earnings and strong growth today."
technical_signal = 1  # Hypothetical technical indicator signal.
signal_ages = [1, 1, 5, 10, 2]  # Example ages for each signal component.
final_signal = aggregate_signals(
    news_text, 
    technical_signal, 
    "AAPL", 
    train_data, 
    train_data['target'], 
    test_data, 
    signal_ages,
    social_query="AAPL"
)
print("Final Aggregated Trading Signal:", final_signal)

# 6. Generate a Dynamic Watchlist using the Asset Selector.
top_assets = select_top_assets(n=10, use_social=True)
print("Dynamic Watchlist:", top_assets)

# 7. Log Outputs.
log_trade(f"ML Predictions: Mean={predictions.mean()}, Std={predictions.std()}")
log_trade(f"Combined Social Sentiment: {social_sentiment}")
log_trade(f"Final Aggregated Signal: {final_signal}")
log_trade(f"Dynamic Watchlist: {top_assets}")

print("Main execution complete.")
