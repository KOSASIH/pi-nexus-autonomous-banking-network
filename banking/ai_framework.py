import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_pass(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward(error)

class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size

    def forward(self, input_data):
        # Convolutional layer forward pass
        pass

    def backward(self, error):
        # Convolutional layer backward pass
        pass

nn = NeuralNetwork([
    ConvolutionalLayer(32, 3),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

input_data = np.random.rand(1, 28, 28, 1)
output = nn.forward_pass(input_data)
print(output.shape)
