import numpy as np
from modules.sentiment_analysis import aggregate_sentiments
from modules.macro_module import get_gdp_indicator
from modules.earnings_module import get_upcoming_earnings
from modules.ml_predictor import ensemble_predict

def temporal_weight(signal, days_since_signal, half_life=5):
    decay_factor = np.exp(-np.log(2) * days_since_signal / half_life)
    return signal * decay_factor

def aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test, signal_ages):
    sentiment = aggregate_sentiments(news_text)
    sentiment_signal = 1 if sentiment['sentiment'] == 'POSITIVE' else -1

    macro_signal = 1 if get_gdp_indicator() == "BULLISH" else -1
    earnings_signal = 1 if get_upcoming_earnings(symbol) == "EVENT_PENDING" else 0

    ml_preds = ensemble_predict(X_train, y_train, X_test)
    ml_signal = 1 if ml_preds.mean() > 0 else -1

    raw_signals = np.array([sentiment_signal, technical_signal, macro_signal, earnings_signal, ml_signal])

    # Apply temporal weighting
    half_life = 5  # adjustable
    weights = np.array([np.exp(-age/half_life) for age in signal_ages])

    final_signal = np.average(raw_signals, weights=weights)
    return final_signal

if __name__ == "__main__":
    news_text = "Company reports record earnings, bullish sentiment prevails."
    technical_signal = 1
    symbol = "AAPL"
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(1, 10)
    signal_ages = [1, 1, 5, 10, 2]  # example ages in days
    
    signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test, signal_ages)
    print("Aggregated Trading Signal:", signal)
