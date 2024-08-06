import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TrainingData:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = self.dataset.drop('target', axis=1)
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

class Trainer:
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def train(self):
        self.model.train(self.X_train, self.y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = self.model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
