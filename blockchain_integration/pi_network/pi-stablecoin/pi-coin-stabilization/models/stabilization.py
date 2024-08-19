import pandas as pd
from ai_models.linear_regression import LinearRegressionModel
from ai_models.decision_trees import DecisionTreeModel
from ai_models.random_forest import RandomForestModel
from ai_models.neural_networks import NeuralNetworkModel
from ai_models.arima import ARIMAModel

class StabilizationModel:
    def __init__(self, data):
        self.data = data
        self.linear_regression_model = LinearRegressionModel(data)
        self.decision_tree_model = DecisionTreeModel(data)
        self.random_forest_model = RandomForestModel(data)
        self.neural_network_model = NeuralNetworkModel(data)
        self.arima_model = ARIMAModel(data)

    def train(self):
        self.linear_regression_model.train()
        self.decision_tree_model.train()
        self.random_forest_model.train()
        self.neural_network_model.train()
        self.arima_model.train()

    def predict(self, input_features):
        predictions = []
        predictions.append(self.linear_regression_model.predict(input_features))
        predictions.append(self.decision_tree_model.predict(input_features))
        predictions.append(self.random_forest_model.predict(input_features))
        predictions.append(self.neural_network_model.predict(input_features))
        predictions.append(self.arima_model.predict(input_features))
        return predictions
