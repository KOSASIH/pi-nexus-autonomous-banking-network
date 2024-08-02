// MachineLearningModel.js

import { NeuralNetwork } from './NeuralNetwork';

class MachineLearningModel {
  constructor() {
    this.neuralNetwork = new NeuralNetwork();
  }

  train(data, labels) {
    // Train the machine learning model using the data and labels
    this.neuralNetwork.train(data, labels);
  }

  predict(input) {
    // Use the machine learning model to make a prediction
    const output = this.neuralNetwork.forward(input);
    return output;
  }
}

export default MachineLearningModel;
