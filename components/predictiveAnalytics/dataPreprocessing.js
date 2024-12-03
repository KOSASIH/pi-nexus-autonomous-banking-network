// components/predictiveAnalytics/dataPreprocessing.js

const { Tensor } = require('@tensorflow/tfjs');

function normalizeData(data) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    return data.map(value => (value - min) / (max - min));
}

function splitData(data, testSize = 0.2) {
    const testCount = Math.floor(data.length * testSize);
    const trainData = data.slice(0, data.length - testCount);
    const testData = data.slice(data.length - testCount);
    return { trainData, testData };
}

function prepareData(data) {
    const normalizedData = normalizeData(data);
    return splitData(normalizedData);
}

module.exports = { prepareData };
