import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class NexusCybersecurityThreatDetectionSystem:
    def __init__(self):
        self.data = pd.read_csv('threat_data.csv')
        self.X = self.data.drop('label', axis=1)
        self.y = self.data['label']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def detect_threats(self, input_data):
        return self.model.predict(input_data)
