const tf = require("@tensorflow/tfjs");

class MachineLearning {
  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 10, inputShape: [784] }));
    this.model.add(tf.layers.dense({ units: 10 }));
    this.model.compile({
      optimizer: tf.optimizers.adam(),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

  train(X, y) {
    this.model.fit(X, y, { epochs: 10 });
  }

  predict(X) {
    return this.model.predict(X);
  }
}

const ml = new MachineLearning();
const X = tf.random.normal([100, 784]);
const y = tf.random.uniform([100, 10]);
ml.train(X, y);
const predictions = ml.predict(X);
console.log(predictions);
