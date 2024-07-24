import numpy as np
from neural_network import NeuralNetwork
from data_loader import DataLoader

def main():
    file_path = 'data.csv'
    data_loader = DataLoader(file_path)
    X, y = data_loader.load_data()
    X, y = data_loader.preprocess_data(X, y)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

    neural_network = NeuralNetwork(input_shape=(28, 28), num_classes=10)
    neural_network.train(X_train, y_train, X_test, y_test, epochs=10)

    loss, accuracy = neural_network.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
