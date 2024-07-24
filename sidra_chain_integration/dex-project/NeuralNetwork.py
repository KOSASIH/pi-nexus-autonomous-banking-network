import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights1 = tf.Variable(tf.random.normal([input_dim, hidden_dim]))
        self.weights2 = tf.Variable(tf.random.normal([hidden_dim, output_dim]))

        self.biases1 = tf.Variable(tf.zeros([hidden_dim]))
        self.biases2 = tf.Variable(tf.zeros([output_dim]))

    def forward(self, x):
        hidden_layer = tf.nn.relu(tf.matmul(x, self.weights1) + self.biases1)
        output_layer = tf.matmul(hidden_layer, self.weights2) + self.biases2
        return output_layer

    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                with tf.GradientTape() as tape:
                    output = self.forward(x_batch)
                    loss = tf.reduce_mean(tf.square(output - y_batch))

                gradients = tape.gradient(loss, [self.weights1, self.weights2, self.biases1, self.biases2])
                optimizer.apply_gradients(zip(gradients, [self.weights1, self.weights2, self.biases1, self.biases2]))

            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

    def predict(self, x):
        return self.forward(x)
