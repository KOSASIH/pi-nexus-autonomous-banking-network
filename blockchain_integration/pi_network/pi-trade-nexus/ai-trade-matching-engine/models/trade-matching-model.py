import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TradeMatchingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Trade Matching Model Training Report:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        print("Trade Matching Model Evaluation Report:")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

# Example usage:
if __name__ == "__main__":
    # Load trade data
    trade_data = pd.read_csv("../data/trade-data.csv")

    # Preprocess data
    X = trade_data.drop(["trade_id", "timestamp"], axis=1)
    y = trade_data["trade_id"]

    # Create and train model
    model = TradeMatchingModel()
    model.train(X, y)

    # Evaluate model
    model.evaluate(X, y)
