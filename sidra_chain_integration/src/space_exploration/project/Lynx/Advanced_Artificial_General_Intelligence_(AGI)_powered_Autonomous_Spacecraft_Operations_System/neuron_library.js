// Neuron library implementation
class Neuron {
  constructor() {
    this.weights = [];
    this.bias = 0;
  }

  setWeights(weights) {
    // Set neuron weights
    this.weights = weights;
  }

  setBias(bias) {
    // Set neuron bias
    this.bias = bias;
  }

  run(data) {
    // Run neuron on input data
    let output = 0;
    for (const weight of this.weights) {
      output += weight * data;
    }
    output += this.bias;
    return output;
  }
}

export default Neuron;
