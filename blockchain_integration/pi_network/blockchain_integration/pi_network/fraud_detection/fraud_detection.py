import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FraudDetection:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data
        self.model = RandomForestClassifier()

    def train_model(self):
        X = self.transaction_data.drop(['label'], axis=1)
        y = self.transaction_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_fraud(self, new_transaction):
        prediction = self.model.predict(new_transaction)
        return prediction

# Example usage:
fraud_detector = FraudDetection(transaction_data)
fraud_detector.train_model()
new_transaction = pd.DataFrame({'amount': [100], 'category': ['withdrawal']})
fraud_prediction = fraud_detector.predict_fraud(new_transaction)
print(fraud_prediction)
