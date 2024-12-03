import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class GAN:
    def __init__(self, noise_dim, data_shape):
        self.noise_dim = noise_dim
        self.data_shape = data_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            layers.Dense(256, activation='relu'),
            layers.Dense(np.prod(self.data_shape), activation='tanh'),
            layers.Reshape(self.data_shape)
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Flatten(input_shape=self.data_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.discriminator.trainable = False
        model = tf.keras.Sequential([self.generator, self.discriminator])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, real_data, epochs=10000, batch_size=128):
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_samples = real_data[idx]
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fake_samples = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 1000 == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")
