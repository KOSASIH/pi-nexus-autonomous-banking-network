# AI/Fraud_Detection_Model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load historical transaction data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    # Convert categorical variables to numerical
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('is_fraud', axis=1)  # Features
    y = data['is_fraud']  # Target variable
    return X, y

# Train the fraud detection model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'fraud_detection_model.pkl')

# Example usage
if __name__ == "__main__":
    data = load_data('transactions.csv')  # Replace with your actual data file
    X, y = preprocess_data(data)
    train_model(X, y)
