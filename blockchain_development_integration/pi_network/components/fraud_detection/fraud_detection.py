# fraud_detection.py
from sklearn.ensemble import IsolationForest

def detect_fraud(transactions):
    model = IsolationForest(contamination=0.01)
    model.fit(transactions[['amount', 'time', 'location']])
    transactions['is_fraud'] = model.predict(transactions[['amount', 'time', 'location']])
    return transactions[transactions['is_fraud'] == -1]
