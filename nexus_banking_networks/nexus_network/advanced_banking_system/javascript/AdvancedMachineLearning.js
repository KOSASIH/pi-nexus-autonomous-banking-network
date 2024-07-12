// AdvancedMachineLearning.js
import * as tf from "@tensorflow/tfjs";

class AdvancedMachineLearning {
  constructor() {
    this.model = this.createModel();
  }

  createModel() {
    const inputLayer = tf.input({ shape: [10] });
    const x = tf.layers
      .dense({ units: 64, activation: "relu" })
      .apply(inputLayer);
    consty = tf.layers.dense({ units: 10, activation: "softmax" }).apply(x);
    const model = tf.model({ inputs: inputLayer, outputs: y });
    model.compile({
      optimizer: "adam",
      loss: "categorical_crossentropy",
      metrics: ["accuracy"],
    });
    return model;
  }

  trainModel(X_train, y_train) {
    return this.model.fit(X_train, y_train, {
      epochs: 10,
      batchSize: 128,
      validationSplit: 0.2,
    });
  }

  makePrediction(inputData) {
    return this.model.predict(inputData);
  }
}
