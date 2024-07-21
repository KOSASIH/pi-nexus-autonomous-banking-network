import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        return self.sia.polarity_scores(text)
