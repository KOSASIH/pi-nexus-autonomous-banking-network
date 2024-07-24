import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=input_shape))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(output_shape, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X_train, y_train, epochs=100):
        self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test):
        return self.model.predict(X_test)
