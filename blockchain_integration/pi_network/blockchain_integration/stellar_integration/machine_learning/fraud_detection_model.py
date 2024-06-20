import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FraudDetectionModel:
    def __init__(self, training_data: pd.DataFrame):
        self.training_data = training_data
        self.model = RandomForestClassifier(n_estimators=100)

    def train_model(self) -> None:
        X = self.training_data.drop(['is_fraud'], axis=1)
        y = self.training_data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_fraud(self, transaction_data: pd.DataFrame) -> bool:
        prediction = self.model.predict(transaction_data)
        return prediction[0]

    def evaluate_model(self) -> float:
        y_pred = self.model.predict(self.training_data.drop(['is_fraud'], axis=1))
        return accuracy_score(self.training_data['is_fraud'], y_pred)
