# ai_engine/ai_model.py
import tensorflow as tf


class AIModel:
    def __init__(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
