# modules/signal_aggregator.py
from modules.sentiment_analysis import aggregate_sentiments
from modules.macro_module import get_gdp_indicator
from modules.earnings_module import get_upcoming_earnings
from modules.ml_predictor import ensemble_predict
import numpy as np

def aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test):
    sentiment = aggregate_sentiments(news_text)
    sentiment_signal = 1 if sentiment['sentiment'] == 'POSITIVE' else -1
    tech_signal = technical_signal
    macro_indicator = get_gdp_indicator()
    
    if macro_indicator == "BULLISH":
        macro_signal = 1
    elif macro_indicator == "BEARISH":
        macro_signal = -1
    else:  # NEUTRAL or unknown
        macro_signal = 0

    earnings_signal = 1 if get_upcoming_earnings(symbol) == "EVENT_PENDING" else 0
    ml_signal = 1 if ensemble_predict(X_train, y_train, X_test).mean() > 0 else -1

    final_signal = np.mean([sentiment_signal, tech_signal, macro_signal, earnings_signal, ml_signal])
    return final_signal
