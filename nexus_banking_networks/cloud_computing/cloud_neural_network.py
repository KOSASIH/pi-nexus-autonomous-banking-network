import tensorflow as tf
from tensorflow import keras

def create_neural_network(input_shape, output_shape):
    # Create a new neural network model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, training_data, validation_data):
    # Train the neural network model
    model.fit(training_data, epochs=10, validation_data=validation_data)
    return model

if __name__ == '__main__':
    input_shape = (784,)
    output_shape = 10

    model = create_neural_network(input_shape, output_shape)
    training_data = ...
    validation_data = ...
    trained_model = train_neural_network(model, training_data, validation_data)
    print("Neural network trained successfully!")
