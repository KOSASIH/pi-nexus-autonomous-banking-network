import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target

    def split_data(self) -> tuple:
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self) -> RandomForestClassifier:
        X_train, X_test, y_train, y_test = self.split_data()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: RandomForestClassifier) -> None:
        X_train, X_test, y_train, y_test = self.split_data()
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
