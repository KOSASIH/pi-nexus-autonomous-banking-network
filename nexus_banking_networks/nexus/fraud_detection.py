import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class FraudDetector:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self):
        X = self.data.drop(['is_fraud'], axis=1)
        y = self.data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, transaction):
        prediction = self.model.predict(transaction)
        return prediction

    def evaluate(self):
        y_pred = self.model.predict(self.data.drop(['is_fraud'], axis=1))
        print("Accuracy:", accuracy_score(self.data['is_fraud'], y_pred))
        print("Classification Report:")
        print(classification_report(self.data['is_fraud'], y_pred))

# Example usage:
data = pd.read_csv('transactions.csv')
detector = FraudDetector(data)
detector.train()
detector.evaluate()
