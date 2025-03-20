# modules/sentiment_analysis.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

hf_sentiment = pipeline("sentiment-analysis")
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_hf(text):
    result = hf_sentiment(text)
    return result[0]

def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)

def aggregate_sentiments(text):
    hf_result = analyze_sentiment_hf(text)
    vader_result = analyze_sentiment_vader(text)
    aggregated = {
        'sentiment': hf_result['label'],
        'score': (hf_result['score'] + vader_result['compound']) / 2
    }
    return aggregated

if __name__ == "__main__":
    sample = "The market is extremely bullish today!"
    print("HuggingFace:", analyze_sentiment_hf(sample))
    print("VADER:", analyze_sentiment_vader(sample))
    print("Aggregated:", aggregate_sentiments(sample))
