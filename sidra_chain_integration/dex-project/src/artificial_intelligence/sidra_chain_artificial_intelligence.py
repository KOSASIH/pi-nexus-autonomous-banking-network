# dex_project_artificial_intelligence.py
import numpy as np
import tensorflow as tf
from tensorflow import keras

class DexProjectArtificialIntelligence:
    def __init__(self):
        pass

    def create_neural_network(self, input_shape, output_shape):
        # Create a neural network using Keras
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(output_shape, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_neural_network(self, model, X_train, y_train, X_test, y_test):
        # Train a neural network using Keras
        model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    def use_neural_network(self, model, input_data):
        # Use a neural network to make predictions
        output_data = model.predict(input_data)
        return output_data

    def create_decision_tree(self, X_train, y_train):
        # Create a decision tree using Scikit-learn
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        return clf

    def use_decision_tree(self, clf, input_data):
        # Use a decision tree to make predictions
        output_data = clf.predict(input_data)
        return output_data
