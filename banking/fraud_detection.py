import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class FraudDetection:
    def __init__(self, data):
        self.data = data
        self.model = self.create_model()

    def create_model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        return model

    def train_model(self):
        X_train, y_train = self.split_data()
        self.model.fit(X_train, y_train)

    def predict_fraud(self, new_data):
        prediction = self.model.predict(new_data)
        return prediction

    def split_data(self):
        X = self.data.drop("fraud", axis=1)
        y = self.data["fraud"]
        return X, y

# Example usage:
data = pd.read_csv("transaction_data.csv")
fraud_detection = FraudDetection(data)
fraud_detection.train_model()
new_data = pd.read_csv("new_transaction_data.csv")
fraud_prediction = fraud_detection.predict_fraud(new_data)
print(fraud_prediction)
