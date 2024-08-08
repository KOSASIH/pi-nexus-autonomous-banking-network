// neural_network.js

const tf = require('@tensorflow/tfjs');
const { Sequential } = require('@tensorflow/tfjs-layers');

class NeuralNetwork {
  constructor() {
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, inputShape: [100] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64 }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.optimizers.adam(), loss: 'meanSquaredError' });
    return model;
  }

  train(data, labels) {
    this.model.fit(data, labels, { epochs: 10, batchSize: 32 });
  }

  predict(data) {
    return this.model.predict(data);
  }
}

module.exports = NeuralNetwork;
