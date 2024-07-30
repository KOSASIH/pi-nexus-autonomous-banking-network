# ai.py
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

class EonixAI:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        # Create a neural network model for predicting transaction outcomes
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, data):
        # Train the model on the given data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        self.model.fit(data_scaled, epochs=10)

    def predict_outcome(self, transaction):
        # Predict the outcome of a transaction using the trained model
        input_data = self.preprocess_transaction(transaction)
        output = self.model.predict(input_data)
        return output

    def preprocess_transaction(self, transaction):
        # Preprocess the transaction data for input into the model
        input_data = [
            transaction.amount,
            transaction.sender,
            transaction.recipient,
            transaction.timestamp,
            # Add more features as needed
        ]
        return input_data
