import numpy as np
import tensorflow as tf
from tensorflow import keras

class NeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
