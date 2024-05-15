# machine_learning/ml_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MLTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        # Train machine learning model
        data = pd.read_csv(self.config.data_file)
        X, y = data.drop('target', axis=1), data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
