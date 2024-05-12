import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RiskAssessor:
    def __init__(self, model_path=None):
        """
        Initializes the RiskAssessor object.

        Args:
            model_path (str, optional): The file path to the trained machine learning model. Defaults to None.
        """
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = self.create_model()

    def load_model(self, model_path):
        """
        Loads the trained machine learning model from the given file path.

        Args:
            model_path (str): The file path to the trained machine learning model.

        Returns:
            A trained machine learning model.
        """
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        model.load_model(model_path)
        return model

    def create_model(self):
        """
        Creates a new machine learning model.

        Returns:
            A new machine learning model.
        """
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        return model

    def train_model(self, X_train, y_train):
        """
        Trains the machine learning model using the given training data.

        Args:
            X_train (pandas.DataFrame): The training data features.
            y_train (pandas.Series): The training data labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the performance of the trained machine learning model using the given testing data.

        Args:
            X_test (pandas.DataFrame): The testing data features.
            y_test (pandas.Series): The testing data labels.

        Returns:
            The AUC score of the trained machine learning model.
        """
        y_pred = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        return auc

    def assess_risk(self, transaction):
        """
        Assesses the risk of a given transaction using the trained machine learning model.

        Args:
            transaction (dict): A dictionary containing the transaction data.

        Returns:
            The risk score of the transaction.
        """
        transaction = pd.DataFrame(
            [transaction], columns=["amount", "category", "merchant"]
        )
        transaction = self.preprocess_data(transaction)
        transaction = self.scaler.transform(transaction)
        risk = self.model.predict_proba(transaction)[:, 1][0]
        return risk

    def preprocess_data(self, data):
        """
        Preprocesses the data by cleaning, transforming, and encoding the features.

        Args:
            data (pandas.DataFrame): The transaction data.

        Returns:
            The preprocessed transaction data.
        """
        # Clean the data
        data = data.dropna()

        # Transform the data
        data["amount"] = np.log(data["amount"])

        # Encode the categorical features
        data = pd.get_dummies(data, columns=["category", "merchant"])

        return data
