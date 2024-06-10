import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class AnomalyDetector:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train an Isolation Forest model
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('label', axis=1), self.data['label'], test_size=0.2, random_state=42)
        if_model = IsolationForest(n_estimators=100, random_state=42)
        if_model.fit(X_train, y_train)
        y_pred = if_model.predict(X_test)
        print("Anomaly Detection Accuracy:", accuracy_score(y_test, y_pred))
        print("Anomaly Detection Classification Report:")
        print(classification_report(y_test, y_pred))

    def detect_anomalies(self, new_data):
        # Use the trained model to detect anomalies in real-time
        new_data = pd.DataFrame(new_data)
        predictions = if_model.predict(new_data)
        return predictions

# Example usage
data = pd.read_csv('transaction_data.csv')
ad = AnomalyDetector(data)
ad.train_model()
new_data = pd.read_csv('new_transactions.csv')
anomalies = ad.detect_anomalies(new_data)
print("Anomalies detected:", anomalies)
