# ai_nlp/models/sentiment.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class SentimentModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
