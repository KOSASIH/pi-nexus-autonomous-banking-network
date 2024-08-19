import pandas as pd
from models.stabilization_model import StabilizationModel

class PricePredictor:
    def __init__(self, data):
        self.data = data
        self.model = StabilizationModel(data)

    def predict(self, input_features):
        return self.model.predict(input_features)
