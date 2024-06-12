import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load transaction data
transactions = pd.read_csv('transactions.csv')

# Preprocess data
X = transactions.drop(['label'], axis=1)
y = transactions['label']

# Train AI model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Use AI model for fraud detection
def detect_fraud(transaction):
    features = extract_features(transaction)
    prediction = rfc.predict(features)
    return prediction
