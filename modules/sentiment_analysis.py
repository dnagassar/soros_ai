# modules/sentiment_analysis.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize the FinBERT model explicitly
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_finbert(text):
    result = finbert(text)[0]
    return {
        'label': result['label'],
        'score': result['score']
    }

def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)

def aggregate_sentiments(text):
    finbert_result = analyze_sentiment_finbert(text)
    vader_result = vader_analyzer.polarity_scores(text)

    # Convert labels to numerical scores for averaging
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    finbert_score = sentiment_mapping.get(finbert_result['label'].lower(), 0) * finbert_result['score']
    vader_score = vader_result['compound']

    # Average scores from both methods
    final_score = np.mean([finbert_score, vader_result['compound']])

    # Determine final sentiment
    if final_score > 0.1:
        final_sentiment = 'POSITIVE'
    elif final_score < -0.1:
        final_sentiment = 'NEGATIVE'
    else:
        final_sentiment = 'NEUTRAL'

    return {
        'sentiment': final_sentiment,
        'score': final_score,
        'individual_scores': {
            'FinBERT': finbert_result,
            'VADER': vader_result
        }
    }

if __name__ == "__main__":
    sample_text = "The company reported record-breaking earnings, causing the stock to surge."
    sentiment_result = aggregate_sentiments(sample_text)
    print("Aggregated Sentiment:", sentiment_result)
