# modules/signal_aggregator.py
import numpy as np
from modules.sentiment_analysis import aggregate_sentiments
from modules.macro_module import get_gdp_indicator
from modules.earnings_module import get_upcoming_earnings
from modules.ml_predictor import ensemble_predict
from modules.news_social_monitor import get_combined_sentiment

def aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test, signal_ages,
                      social_query=None):
    """
    Aggregates multiple signals into a final trading signal.
    
    Parameters:
      - news_text: A traditional news headline.
      - technical_signal: Signal from technical indicators (e.g., 1 for buy, -1 for sell).
      - symbol: The asset ticker.
      - X_train, y_train, X_test: Data for the ML ensemble prediction.
      - signal_ages: List of ages (in days) for each signal component.
      - social_query: If provided, used to fetch combined social sentiment.
    
    Returns:
      - final_signal: A float representing the combined trading signal.
    """
    # Traditional news sentiment
    base_sentiment = aggregate_sentiments(news_text)
    base_score = base_sentiment.get("score", 0)
    base_label = base_sentiment.get("sentiment", "NEUTRAL")
    base_signal = 1 if base_label == "POSITIVE" else -1 if base_label == "NEGATIVE" else 0
    
    # Social sentiment from Reddit/StockTwits/RSS feeds, if requested
    if social_query:
        social_data = get_combined_sentiment(social_query, symbol=symbol)
        social_score = social_data.get("score", 0)
        # Combine with weighting: 60% traditional, 40% social
        combined_score = (0.6 * base_score) + (0.4 * social_score)
        combined_signal = 1 if combined_score > 0.5 else -1 if combined_score < -0.5 else 0
    else:
        combined_signal = base_signal

    # Macro signal
    macro_signal = 1 if get_gdp_indicator() == "BULLISH" else -1
    # Earnings signal
    earnings_signal = 1 if get_upcoming_earnings(symbol) == "EVENT_PENDING" else 0
    # ML signal
    ml_preds = ensemble_predict(X_train, y_train, X_test)
    ml_signal = 1 if ml_preds.mean() > 0 else -1

    # Combine all signals with temporal weighting
    raw_signals = np.array([combined_signal, technical_signal, macro_signal, earnings_signal, ml_signal])
    half_life = 5  # Adjustable parameter
    weights = np.array([np.exp(-age / half_life) for age in signal_ages])
    final_signal = np.average(raw_signals, weights=weights)
    return final_signal

if __name__ == "__main__":
    news_text = "Company reports record earnings, bullish sentiment prevails."
    technical_signal = 1
    symbol = "AAPL"
    import numpy as np
    # Create dummy ML data
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(1, 10)
    signal_ages = [1, 1, 5, 10, 2]
    final_signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test,
                                     signal_ages, social_query="AAPL")
    print("Final Aggregated Trading Signal:", final_signal)
