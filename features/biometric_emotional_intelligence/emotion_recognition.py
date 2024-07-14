# File name: emotion_recognition.py
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class EmotionRecognition:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(7, activation="softmax"))

    def train(self, X, y):
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(X, y, epochs=10, batch_size=32)

    def recognize(self, image):
        return self.model.predict(image)
