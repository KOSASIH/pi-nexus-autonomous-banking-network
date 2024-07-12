import * as tf from '@tensorflow/tfjs';

class AI {
  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 10, inputShape: [1] }));
    this.model.add(tf.layers.dense({ units: 1 }));
    this.model.compile({ optimizer: tf.optimizers.adam(), loss: 'meanSquaredError' });
  }

  train(data) {
    this.model.fit(data, { epochs: 100 });
  }

  predict(input) {
    return this.model.predict(input);
  }
}

const ai = new AI();

// Example usage:
const data = [
  { x: 10, y: 20 },
  { x: 20, y: 40 },
  { x: 30, y: 60 },
  { x: 40, y: 80 },
  { x: 50, y: 100 }
];

ai.train(data);
const prediction = ai.predict(tf.tensor2d([60], [1, 1]));
console.log(prediction.dataSync());
