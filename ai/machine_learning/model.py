import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class DecisionTreeRegressionModel:
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class RandomForestRegressionModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class NeuralNetworkRegressionModel:
    def __init__(self):
        self.model = MLPRegressor()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
