# ml.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class EonixML:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        # Create a random forest classifier for identifying fraudulent transactions
        model = RandomForestClassifier(n_estimators=100)
        return model

    def train_model(self, data):
        # Train the model on the given data
        self.model.fit(data.drop('label', axis=1), data['label'])

    def predict_fraud(self, transaction):
        # Predict whether a transaction is fraudulent using the trained model
        input_data = self.preprocess_transaction(transaction)
        output = self.model.predict(input_data)
        return output

    def preprocess_transaction(self, transaction):
        # Preprocess the transaction data for input into the model
        input_data = pd.DataFrame([
            transaction.amount,
            transaction.sender,
            transaction.recipient,
            transaction.timestamp,
            # Add more features as needed
        ]).T
        return input_data
