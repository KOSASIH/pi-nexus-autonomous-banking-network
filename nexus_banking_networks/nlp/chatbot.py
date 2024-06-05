import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class Chatbot:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train NLP model on historical data
        X = self.data['text']
        y = self.data['response']
        vectorizer = TfidfVectorizer()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(vectorizer.fit_transform(X_train), y_train)
        return model

    def respond_to_user(self, model):
        # Respond to user input using trained model
        user_input = 'What is the weather like today?'
        vectorized_input = vectorizer.transform([user_input])
        response = model.predict(vectorized_input)
        return response
