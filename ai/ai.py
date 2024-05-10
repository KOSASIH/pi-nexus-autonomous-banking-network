import numpy as np
import tensorflow as tf
from tensorflow import keras


class AI:
    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_layers=None,
        activation_function=tf.nn.relu,
    ):
        """
        Initializes the AI class with the specified input and output shapes, and optional hidden layer sizes and activation function.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function

        self.model = None

    def build_model(self):
        """
        Builds the AI model using the specified input and output shapes, hidden layer sizes, and activation function.
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        if self.hidden_layers is not None:
            for hidden_layer in self.hidden_layers:
                x = keras.layers.Dense(
                    hidden_layer, activation=self.activation_function
                )(x)

        outputs = keras.layers.Dense(self.output_shape)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """
        Trains the AI model using the specified training data, number of epochs, and batch size.
        """
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        """
        Makes predictions using the AI model for the specified input data.
        """
        return self.model.predict(x)
