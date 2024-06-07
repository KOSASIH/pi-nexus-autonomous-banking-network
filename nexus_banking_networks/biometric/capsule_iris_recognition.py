import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import CapsuleLayer

class CapsuleIRISRecognizer:
    def __init__(self, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(CapsuleLayer(num_capsules=10, dim_capsule=16, routings=3))
        self.model.add(Dense(num_classes, activation='softmax'))

    def recognize_iris(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        return prediction

# Example usage
recognizer = CapsuleIRISRecognizer(num_classes=10)
image_path = 'iris_image.jpg'
prediction = recognizer.recognize_iris(image_path)
print(f'Prediction: {prediction}')
