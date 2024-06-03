import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score


class WalletMLModelTrainer:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data

    def train_anomaly_detection_model(self):
        # Extract features from transaction data
        features = self.transaction_data[["amount", "category", "location", "time"]]

        # Train isolation forest model
        model = IsolationForest(contamination=0.1)
        model.fit(features)

        return model

    def train_user_behavior_model(self):
        # Extract features from transaction data
        features = self.transaction_data[["amount", "category", "location", "time"]]
        labels = self.transaction_data["user_id"]

        # Train random forest classifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(features, labels)

        return model


if __name__ == "__main__":
    transaction_data = pd.read_csv("transaction_data.csv")
    wallet_ml_model_trainer = WalletMLModelTrainer(transaction_data)

    anomaly_detection_model = wallet_ml_model_trainer.train_anomaly_detection_model()
    user_behavior_model = wallet_ml_model_trainer.train_user_behavior_model()

    # Use trained models to predict anomalies and user behavior
    anomaly_predictions = anomaly_detection_model.predict(transaction_data)
    user_behavior_predictions = user_behavior_model.predict(transaction_data)

    print("Anomaly Predictions:")
    print(anomaly_predictions)

    print("User Behavior Predictions:")
    print(user_behavior_predictions)
