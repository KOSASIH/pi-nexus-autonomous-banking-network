// index.js
const { preprocessData } = require('./dataPreprocessing');
const { trainModel, predict } = require('./model');
const { evaluateModel } = require('./modelEvaluation');

async function runPredictiveAnalytics(rawData) {
    // Step 1: Preprocess the data
    const processedData = preprocessData(rawData);

    // Step 2: Train the model
    const model = await trainModel(processedData);

    // Step 3: Evaluate the model
    const evaluationResults = evaluateModel(model, processedData);

    return {
        model,
        evaluationResults,
    };
}

async function makePrediction(model, newData) {
    const prediction = await predict(model, newData);
    return prediction;
}

module.exports = { runPredictiveAnalytics, makePrediction };
