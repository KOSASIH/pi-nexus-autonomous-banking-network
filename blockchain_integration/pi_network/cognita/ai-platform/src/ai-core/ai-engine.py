import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class AIEngine:
    def __init__(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(784,)),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])

    def train(self, X_train, y_train):
        # Training logic
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10)

    def predict(self, X_test):
        # Prediction logic
        return self.model.predict(X_test)
