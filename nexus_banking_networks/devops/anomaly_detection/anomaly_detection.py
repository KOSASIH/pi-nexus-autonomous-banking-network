import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load transaction data
transactions = pd.read_csv('transactions.csv')

# Create an Isolation Forest model for anomaly detection
if_model = IsolationForest(contamination=0.1)
if_model.fit(transactions)

# Create an LSTM model for time-series forecasting
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(transactions.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Integrate both models for hybrid anomaly detection
def detect_anomalies(transaction):
    if_score = if_model.decision_function([transaction])
    lstm_pred = lstm_model.predict([transaction])
    if if_score < 0.5 and lstm_pred > 2 * transactions.std():
        return True
    return False

# Test the anomaly detection system
test_transactions = pd.read_csv('test_transactions.csv')
anomaly_detected = [detect_anomalies(row) for row in test_transactions.values]
print("Anomaly detection accuracy:", accuracy_score(test_transactions['label'], anomaly_detected))
