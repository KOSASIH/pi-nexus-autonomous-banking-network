import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class RiskAssessment:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train random forest classifier model on historical data
        X = self.data.drop(['target'], axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def assess_risk(self, model):
        # Assess risk of new transactions using trained model
        predictions = model.predict(self.data)
        return predictions
