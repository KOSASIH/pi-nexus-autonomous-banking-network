import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

class RTFraudDetector:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = IsolationForest(contamination=0.1)

    def train_model(self):
        self.model.fit(self.dataset)

    def predict_fraud(self, transaction_data):
        prediction = self.model.predict(transaction_data)
        return prediction

    def update_model(self, new_data):
        self.dataset = pd.concat([self.dataset, new_data])
        self.train_model()

# Example usage:
fraud_detector = RTFraudDetector(pd.read_csv('fraud_data.csv'))
fraud_detector.train_model()

# Predict fraud for a new transaction
transaction_data = pd.DataFrame({'amount': [500], 'category': ['online_payment']})
fraud_prediction = fraud_detector.predict_fraud(transaction_data)
print(f'Fraud prediction: {fraud_prediction}')
