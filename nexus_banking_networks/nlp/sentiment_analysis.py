import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class SentimentAnalysis:

    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train NLP model on historical data
        X = self.data["text"]
        y = self.data["sentiment"]
        vectorizer = TfidfVectorizer()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = MultinomialNB()
        model.fit(vectorizer.fit_transform(X_train), y_train)
        return model

    def analyze_sentiment(self, model):
        # Analyze sentiment of new text data using trained model
        text = ["This is a positive review"]
        vectorized_text = vectorizer.transform(text)
        sentiment = model.predict(vectorized_text)
        return sentiment
