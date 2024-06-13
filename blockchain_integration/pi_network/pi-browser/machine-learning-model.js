import { TensorFlow } from 'tensorflow';

class MachineLearningModel {
  constructor() {
    this.tf = new TensorFlow();
  }

  async trainModel(data) {
    const model = await this.tf.trainModel(data);
    return model;
  }

  async makePrediction(input) {
    const prediction = await this.tf.makePrediction(input);
    return prediction;
  }
}

export default MachineLearningModel;
