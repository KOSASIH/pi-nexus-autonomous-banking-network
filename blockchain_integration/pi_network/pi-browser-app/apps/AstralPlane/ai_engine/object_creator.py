# object_creator.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

class ObjectCreator:
    def __init__(self):
        self.gan = self.build_gan()

    def build_gan(self):
        # Generator network
        generator_input = Input(shape=(100,))
        x = Dense(128, activation='relu')(generator_input)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(3*3*3, activation='tanh')(x)
        generator_output = Reshape((3, 3, 3))(x)

        # Discriminator network
        discriminator_input = Input(shape=(3, 3, 3))
        x = Flatten()(discriminator_input)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid')(x)
        discriminator_output = x

        # GAN model
        gan_input = Input(shape=(100,))
        gan_output = discriminator_output(generator_output)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def generate_object(self, noise):
        return self.gan.predict(noise)
