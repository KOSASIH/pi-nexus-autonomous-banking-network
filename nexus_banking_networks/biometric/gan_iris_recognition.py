import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Input, Reshape, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose

class GANIRISRecognizer:
    def __init__(self, generator_path, discriminator_path):
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)

    def recognize_iris(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        generated_img = self.generator.predict(img)
        prediction = self.discriminator.predict(generated_img)
        return prediction

# Example usage
recognizer = GANIRISRecognizer('generator.h5', 'discriminator.h5')
image_path = 'iris_image.jpg'
prediction = recognizer.recognize_iris(image_path)
print(f'Prediction: {prediction}')
