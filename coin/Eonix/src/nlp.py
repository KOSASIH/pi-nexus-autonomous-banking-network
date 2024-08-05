# nlp.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

class EonixNLP:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        # Analyze the sentiment of a given text
        sentiment = self.sia.polarity_scores(text)
        return sentiment

    def extract_entities(self, text):
        # Extract entities from a given text
        entities = []
        tokens = word_tokenize(text)
        for token in tokens:
            if token.isupper():
                entities.append(token)
        return entities
