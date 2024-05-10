import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class MachineLearning:
    def __init__(self):
        pass

    def train_model(self, data):
        X = data[['feature1', 'feature2']]
        y = data['target']
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict(self, model, data):
        X = data[['feature1', 'feature2']]
        y = model.predict(X)
        return y
