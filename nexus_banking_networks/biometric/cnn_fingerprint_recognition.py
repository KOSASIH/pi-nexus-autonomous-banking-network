import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNNFingerprintRecognizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def recognize_fingerprint(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        return prediction

# Example usage
recognizer = CNNFingerprintRecognizer('fingerprint_model.h5')
image_path = 'fingerprint_image.jpg'
prediction = recognizer.recognize_fingerprint(image_path)
print(f'Prediction: {prediction}')
