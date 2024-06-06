import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class AIFD:
    def __init__(self, training_data_path):
        self.training_data = pd.read_csv(training_data_path)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        X = self.training_data.drop(['label'], axis=1)
        y = self.training_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, transaction_data):
        prediction = self.model.predict(transaction_data)
        return prediction

    def evaluate(self, transaction_data, labels):
        predictions = self.model.predict(transaction_data)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        return accuracy, report
