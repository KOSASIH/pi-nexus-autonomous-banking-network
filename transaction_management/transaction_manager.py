import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TransactionManager:
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def preprocess_transaction(self, transaction):
        """
        Preprocesses the transaction data by cleaning, transforming, and encoding the features.
        """
        transaction = pd.DataFrame([transaction], columns=['amount', 'category', 'merchant'])
        transaction = self.preprocess_transaction_data(transaction)
        transaction = self.scaler.transform(transaction)
        return transaction

    def assess_transaction_risk(self, transaction):
        """
        Assesses the risk of a given transaction using the trained transaction machine learning model.
        """
        transaction = self.preprocess_transaction(transaction)risk = self.model.predict_proba(transaction)[:, 1][0]
        return risk

    def preprocess_transaction_data(self, data):
        """
        Preprocesses the transaction data by cleaning, transforming, and encoding the features.
        """
        # Clean the data
        data = data.dropna()

        # Transform the data
        data['amount'] = np.log(data['amount'])

        # Encode the categorical features
        data = pd.get_dummies(data, columns=['category', 'merchant'])

        return data

    def manage_transaction(self, transaction):
        """
        Manages the transaction based on the assessed risk.
        """
        risk = self.assess_transaction_risk(transaction)
        if risk > 0.5:
            # Decline the transaction
            return 'Decline'
        else:
            # Approve the transaction
            return 'Approve'
