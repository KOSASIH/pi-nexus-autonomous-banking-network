import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


class NaturalLanguageProcessing:

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        sentiment = self.sia.polarity_scores(text)
        return sentiment


nlp = NaturalLanguageProcessing()
text = "I love this product!"
sentiment = nlp.analyze_sentiment(text)
print(sentiment)
