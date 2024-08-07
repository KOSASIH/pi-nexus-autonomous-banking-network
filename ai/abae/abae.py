import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ABAE:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, data):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, data):
        return self.model.predict(data)
