# automl/model_training.py
import joblib

from .model_selection import ModelSelection


class ModelTraining:
    def __init__(self):
        self.model_selection = ModelSelection()

    def train_model(self, X, y):
        model = self.model_selection.select_model(X, y)
        model.fit(X, y)
        return model
