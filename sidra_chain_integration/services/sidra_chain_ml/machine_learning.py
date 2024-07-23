import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class MachineLearning:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, transaction: dict):
        # Train machine learning model here
        pass
