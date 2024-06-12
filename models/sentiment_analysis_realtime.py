import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up sentiment analysis model
sia = SentimentIntensityAnalyzer()

# Analyze sentiment in real-time
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment
