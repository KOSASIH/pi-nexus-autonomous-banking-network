// NeuralNetwork.js

import { Tensor } from 'tensor';

class NeuralNetwork {
  constructor() {
    this.layers = [];
    this.weights = [];
    this.biases = [];
  }

  addLayer(layer) {
    this.layers.push(layer);
  }

  train(data, labels) {
    // Train the neural network using backpropagation
    for (let i = 0; i < data.length; i++) {
      const input = data[i];
      const output = labels[i];
      let error = 0;
      for (let j = 0; j < this.layers.length; j++) {
        const layer = this.layers[j];
        const weights = this.weights[j];
        const biases = this.biases[j];
        const outputLayer = layer.forward(input, weights, biases);
        error += this.calculateError(output, outputLayer);
        input = outputLayer;
      }
      this.backpropagate(error);
    }
  }

  calculateError(output, outputLayer) {
    // Calculate the error between the predicted output and the actual output
    return Math.pow(output - outputLayer, 2);
  }

  backpropagate(error) {
    // Update the weights and biases using backpropagation
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];
      const weights = this.weights[i];
      const biases = this.biases[i];
      layer.backward(error, weights, biases);
    }
  }
}

export default NeuralNetwork;
