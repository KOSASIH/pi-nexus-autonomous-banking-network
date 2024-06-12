import pandas as pd
from transformers import pipeline

class SentimentAnalysis:
    def __init__(self, model_name):
        self.model = pipeline('sentiment-analysis', model=model_name)

    def analyze_sentiment(self, text):
        result = self.model(text)
        return result

# Example usage:
sentiment_analyzer = SentimentAnalysis('distilbert-base-uncased-finetuned-sst-2-english')
text = 'I love the new PI-Nexus Autonomous Banking Network!'
result = sentiment_analyzer.analyze_sentiment(text)
print(result)
