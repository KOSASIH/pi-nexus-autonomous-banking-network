import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class FraudDetector:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()

    def train_model(self):
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        self.model.fit(X, y)

    def predict(self, new_data):
        return self.model.predict(new_data)

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('fraud_data.csv')  # Load fraud detection data
    detector = FraudDetector(data)
    detector.train_model()
    new_data = pd.read_csv('new_transactions.csv')  # Load new transactions
    predictions = detector.predict(new_data)
    print("Fraud Predictions:", predictions)
