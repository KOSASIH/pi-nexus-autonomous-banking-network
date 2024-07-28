// Neural network library implementation
class NeuralNetwork {
  constructor() {
    this.layers = [];
  }

  addLayer(layer) {
    // Add layer to neural network
    this.layers.push(layer);
  }

  run(data) {
    // Run neural network on input data
    let output = data;
    for (const layer of this.layers) {
      output = layer.run(output);
    }
    return output;
  }
}

export default NeuralNetwork;
