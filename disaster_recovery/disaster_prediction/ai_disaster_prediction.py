import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("disaster_data.csv")

# Preprocess data


def preprocess_data(data):
    """Preprocess the disaster data."""
    # Fill missing values
    data.fillna(method="ffill", inplace=True)
    data.fillna(method="bfill", inplace=True)
    data.fillna(0, inplace=True)
    # Scale numerical data
    data[["temperature", "humidity", "pressure"]] = data[
        ["temperature", "humidity", "pressure"]
    ].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # One-hot encode categorical data
    data = pd.get_dummies(data, columns=["disaster_type"])
    return data


data = preprocess_data(data)

# Split data into features and labels
X = data.drop("disaster", axis=1)
y = data["disaster"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation="softmax"),
    ]
)

# Compile model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("disaster_prediction_model.h5")
