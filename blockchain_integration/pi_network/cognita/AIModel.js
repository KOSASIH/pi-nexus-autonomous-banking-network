import * as tf from '@tensorflow/tfjs';
import { BrainJS } from 'brain.js';

class AIModel {
  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 10, inputShape: [10] }));
    this.model.add(tf.layers.dense({ units: 10 }));
    this.model.compile({ optimizer: tf.optimizers.adam(), loss: 'eanSquaredError' });

    this.brain = new BrainJS.NeuralNetwork();
    this.brain.fromJSON({
      "layers": [
        {"type": "input", "neurons": 10},
        {"type": "hidden", "neurons": 10},
        {"type": "output", "neurons": 10}
      ]
    });
  }

  train(data) {
    this.model.fit(data, { epochs: 10 });
    this.brain.train(data);
  }

  predict(input) {
    return this.model.predict(input);
  }

  think(input) {
    return this.brain.run(input);
  }
}

export default AIModel;
