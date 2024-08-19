import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentPredictor:
    def __init__(self, data):
        self.data = data
        self.sia = SentimentIntensityAnalyzer()

    def predict(self, input_text):
        sentiment = self.sia.polarity_scores(input_text)
        return sentiment['compound']
