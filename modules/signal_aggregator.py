import numpy as np
from modules.sentiment_analysis import aggregate_sentiments
from modules.macro_module import get_gdp_indicator
from modules.earnings_module import get_upcoming_earnings
from modules.ml_predictor import ensemble_predict

def aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test):
    # Sentiment Signal
    sentiment = aggregate_sentiments(news_text)
    sentiment_signal = 1 if sentiment['sentiment'] == 'POSITIVE' else -1

    # Technical Signal (provided externally)
    tech_signal = technical_signal

    # Macro Signal
    macro_signal = 1 if get_gdp_indicator() == "BULLISH" else -1

    # Earnings Signal
    earnings_signal = 1 if get_upcoming_earnings(symbol) == "EVENT_PENDING" else 0

    # ML Signal
    ml_preds = ensemble_predict(X_train, y_train, X_test)
    ml_signal = 1 if ml_preds.mean() > 0 else -1

    # Aggregate all signals (simple average)
    final_signal = np.mean([sentiment_signal, tech_signal, macro_signal, earnings_signal, ml_signal])
    return final_signal

if __name__ == "__main__":
    news_text = "Company reports record earnings, bullish sentiment prevails."
    technical_signal = 1
    symbol = "AAPL"
    import numpy as np
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(1, 10)
    
    signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test)
    print("Aggregated Trading Signal:", signal)
