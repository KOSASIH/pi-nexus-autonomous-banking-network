import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class BankingSystem:

    def __init__(self):
        self.data = pd.read_csv("banking_data.csv")

    def train_model(self):
        X = self.data.drop("target", axis=1)
        y = self.data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))

    def make_prediction(self, input_data):
        model = self.train_model()
        prediction = model.predict(input_data)
        return prediction


banking_system = BankingSystem()
banking_system.train_model()
