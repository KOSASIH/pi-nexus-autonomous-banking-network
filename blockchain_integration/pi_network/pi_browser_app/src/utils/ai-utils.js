import * as tf from '@tensorflow/tfjs';

const trainModel = async () => {
  const model = tf.sequential();
  // Add layers to the model
  model.compile({ optimizer: tf.optimizers.adam(), loss: 'eanSquaredError' });
  const data = await fetch('/api/data');
  const dataArray = await data.json();
  model.fit(dataArray, { epochs: 10 });
  return model;
};

const predict = async (inputData) => {
  const model = await trainModel();
  const output = model.predict(inputData);
  return output.dataSync();
};

export { trainModel, predict };
