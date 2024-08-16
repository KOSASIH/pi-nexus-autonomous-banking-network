import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, dataset):
        self.model.fit(dataset.drop('target', axis=1), dataset['target'])

    def predict(self, transaction):
        return self.model.predict(transaction)
