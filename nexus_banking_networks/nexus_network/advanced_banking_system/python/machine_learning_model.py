# machine_learning_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class MachineLearningModel:
    def __init__(self):
        self.data = pd.read_csv('banking_data.csv')

    def train_model(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def make_prediction(self, input_data):
        model = self.train_model()
        prediction = model.predict(input_data)
        return prediction
