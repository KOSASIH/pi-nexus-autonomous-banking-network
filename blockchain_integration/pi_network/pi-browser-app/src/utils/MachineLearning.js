import * as tf from '@tensorflow/tfjs';

class MachineLearning {
  constructor() {
    this.model = null;
  }

  async loadModel() {
    this.model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/autonomous-vehicle-model.tgz');
  }

  async predict(input) {
    const output = this.model.predict(input);
    return output.dataSync();
  }

  async train(data) {
    const xs = data.map((item) => item.input);
    const ys = data.map((item) => item.output);
    this.model.fit(xs, ys, { epochs: 10 });
  }
}

export default MachineLearning;
