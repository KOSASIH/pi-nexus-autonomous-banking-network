# ai_nlp/models/text_classification.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class TextClassificationModel:
    def __init__(self):
        self.model = SVC()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
