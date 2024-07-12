# neural_network_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class NeuralNetworkModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(10,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    def make_prediction(self, input_data):
        return self.model.predict(input_data)
