# neural_architecture_search.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class NeuralArchitectureSearch:
    def __init__(self):
        self.search_space = self.create_search_space()

    def create_search_space(self):
        search_space = {
            'layers': [
                {'type': 'dense', 'units': 64, 'activation': 'elu'},
                {'type': 'dense', 'units': 128, 'activation': 'elu'},
                {'type': 'dense', 'units': 256, 'activation': 'elu'}
            ],
            'connections': [
                {'from': 0, 'to': 1},
                {'from': 0, 'to': 2},
                {'from': 1, 'to': 2}
            ]
        }
        return search_space

    def search(self, data):
        best_model = None
        best_loss = float('inf')
        for i in range(100):
            model = self.sample_model()
            loss = self.evaluate_model(model, data)
            if loss < best_loss:
                best_model = model
                best_loss = loss
        return best_model

    def sample_model(self):
        model = keras.Sequential()
        for layer in self.search_space['layers']:
            if layer['type'] == 'dense':
                model.add(layers.Dense(layer['units'], activation=layer['activation']))
        return model

    def evaluate_model(self, model, data):
        loss = model.evaluate(data)
        return loss
