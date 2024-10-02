# natural_language_processing.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


class NaturalLanguageProcessing:

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def tokenize_text(self, text):
        return word_tokenize(text)
