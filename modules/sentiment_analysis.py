# modules/sentiment_analysis.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Explicitly specify the model and revision for production readiness
hf_sentiment = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    revision="main"
)

# Initialize VADER sentiment analyzer
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
    print("HuggingFace Result:", analyze_sentiment_hf(sample))
    print("VADER Result:", analyze_sentiment_vader(sample))
  
