import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictiveModel:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        X = self.data.drop(['target'], axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def make_prediction(self, model, input_data):
        return model.predict(input_data)

    def evaluate_model(self, model, X_test, y_test):
        accuracy = model.score(X_test, y_test)
        return accuracy
