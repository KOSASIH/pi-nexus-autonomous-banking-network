# Importing necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Class for natural language processing
class PiNetworkNaturalLanguageProcessing:
    def __init__(self):
        self.tokenizer = word_tokenize
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    # Function to tokenize text
    def tokenize_text(self, text):
        return self.tokenizer(text)

    # Function to analyze sentiment
    def analyze_sentiment(self, text):
        return self.sentiment_analyzer.polarity_scores(text)

# Example usage
nlp = PiNetworkNaturalLanguageProcessing()
text = "I love Pi Network!"
tokens = nlp.tokenize_text(text)
sentiment = nlp.analyze_sentiment(text)
print(tokens)
print(sentiment)
