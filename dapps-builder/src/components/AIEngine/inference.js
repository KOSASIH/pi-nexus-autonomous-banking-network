export const makePrediction = async (model, input) => {
    const inputTensor = tf.tensor2d([input]);
    const prediction = model.predict(inputTensor);
    return prediction.arraySync();
};
