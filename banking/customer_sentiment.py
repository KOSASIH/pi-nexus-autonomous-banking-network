import pandas as pd
import numpy as np
from textblob import TextBlob

class CustomerSentiment:
    def __init__(self, data):
        self.data = data

    def analyze_sentiment(self):
        sentiment_scores = []
        for text in self.data["feedback"]:
            sentiment = TextBlob(text)
            sentiment_scores.append(sentiment.sentiment.polarity)
        self.data["sentiment_score"] = sentiment_scores

    def get_sentiment_summary(self):
        avg_sentiment = np.mean(self.data["sentiment_score"])
        if avg_sentiment > 0:
            return "Positive"
        elif avg_sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

# Example usage:
data = pd.read_csv("customer_feedback.csv")
customer_sentiment = CustomerSentiment(data)
customer_sentiment.analyze_sentiment()
sentiment_summary = customer_sentiment.get_sentiment_summary()
print(sentiment_summary)
