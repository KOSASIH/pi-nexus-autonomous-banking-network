import * as tf from '@tensorflow/tfjs';

export const trainModel = async (data) => {
    const { inputs, labels } = data;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputs[0].length] }));
    model.add(tf.layers.dense({ units: labels[0].length, activation: 'softmax' }));

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(labels);

    await model.fit(xs, ys, { epochs: 100 });
    return model;
};

export const saveModel = async (model, path) => {
    await model.save(`file://${path}`);
};

export const loadModel = async (path) => {
    return await tf.loadLayersModel(`file://${path}/model.json`);
};
