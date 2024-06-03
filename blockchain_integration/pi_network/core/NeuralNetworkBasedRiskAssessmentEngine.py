import json
import os

import tensorflow as tf
from kafka import KafkaConsumer
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class NeuralNetworkBasedRiskAssessmentEngine:
    def __init__(self, kafka_topic, blockchain_network):
        self.kafka_consumer = KafkaConsumer(kafka_topic)
        self.blockchain_network = blockchain_network

    def real_time_risk_assessment(self):
        # Analyze real-time data from the PI-Nexus network
        data = self.kafka_consumer.poll(1000)
        X, y = self.preprocess_data(data)

        # Define the neural network architecture
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=3, activation="relu", input_shape=(10, 1)
                ),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(
            X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
        )

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        # Visualize the risk metrics
        self.visualize_risk_metrics(y_test, y_pred)

    def preprocess_data(self, data):
        # Preprocess the data
        X = []
        y = []

        for item in data:
            X.append(item["data"])
            y.append(item["label"])

        return X, y

    def visualize_risk_metrics(self, y_test, y_pred):
        # Visualize the risk metrics using Matplotlib
        plt.plot(y_test, label="Actual Risk")
        plt.plot(y_pred, label="Predicted Risk")
        plt.xlabel("Time")
        plt.ylabel("Risk Score")
        plt.title("Risk Assessment Performance")
        plt.legend()
        plt.show()

    def run(self):
        while True:
            self.real_time_risk_assessment()
