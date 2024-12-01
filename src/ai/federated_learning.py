import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# Define a simple Keras model for federated learning
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),  # Example input shape for MNIST
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Convert the Keras model to a TFF model
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

class FederatedLearning:
    def __init__(self):
        self.federated_data = None
        self.state = None

    def create_federated_data(self, client_data):
        """Create federated data from client datasets."""
        self.federated_data = [tff.simulation.ClientData.from_clients_and_fn(
            client_ids=[str(i)],
            create_client_data_fn=lambda: client_data[i]
        ) for i in range(len(client_data))]

    def train(self, rounds=10):
        """Train the model using federated learning."""
        federated_averaging = tff.learning.build_federated_averaging_process(model_fn)
        self.state = federated_averaging.initialize()

        for round_num in range(1, rounds + 1):
            self.state, metrics = federated_averaging.next(self.state, self.federated_data)
            print(f'Round {round_num}, Metrics: {metrics}')

    def evaluate(self):
        """Evaluate the model on federated data."""
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        metrics = evaluation(self.state.model, self.federated_data)
        print(f'Evaluation Metrics: {metrics}')

# Example usage
if __name__ == "__main__":
    # Simulate some client data
    num_clients = 5
    client_data = [
        tf.data.Dataset.from_tensor_slices((
            np.random.rand(100, 784).astype(np.float32),  # Features
            np.random.randint(0, 10, size=(100,)).astype(np.int32)  # Labels
        )).batch(10) for _ in range(num_clients)
    ]

    federated_learning = FederatedLearning()
    federated_learning.create_federated_data(client_data)
    federated_learning.train(rounds=10)
    federated_learning.evaluate()
