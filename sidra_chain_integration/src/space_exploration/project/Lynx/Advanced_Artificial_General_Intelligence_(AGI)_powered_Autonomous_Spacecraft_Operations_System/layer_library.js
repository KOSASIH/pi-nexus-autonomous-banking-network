// Layer library implementation
class Layer {
  constructor() {
    this.neurons = [];
  }

  addNeuron(neuron) {
    // Add neuron to layer
    this.neurons.push(neuron);
  }

  run(data) {
    // Run layer on input data
    let output = [];
    for (const neuron of this.neurons) {
      output.push(neuron.run(data));
    }
    return output;
  }
}

export default Layer;
