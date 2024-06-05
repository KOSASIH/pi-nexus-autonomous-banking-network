import cv2
import numpy as np
from sklearn.svm import SVC


class ImageRecognition:

    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train computer vision model on historical data
        X = self.data["images"]
        y = self.data["labels"]
        model = SVC(kernel="linear", random_state=42)
        model.fit(X, y)
        return model

    def recognize_image(self, model):
        # Recognize new image using trained model
        image = cv2.imread("image.jpg")
        image = cv2.resize(image, (224, 224))
        image = image.reshape((1, 224, 224, 3))
        prediction = model.predict(image)
        return prediction
