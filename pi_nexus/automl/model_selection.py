# automl/model_selection.py
import pandas as pd
from sklearn.model_selection import train_test_split


class ModelSelection:
    def __init__(self):
        self.data_handler = DataHandler()

    def select_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # implementation
        pass
