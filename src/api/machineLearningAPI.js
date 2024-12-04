// machineLearningAPI.js

import * as tf from '@tensorflow/tfjs';

class MachineLearningAPI {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.model = null;
    }

    // Load the machine learning model
    async loadModel() {
        try {
            this.model = await tf.loadLayersModel(this.modelPath);
            console.log('Model loaded successfully.');
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }

    // Preprocess input data
    preprocessData(inputData) {
        // Convert input data to a tensor
        const tensorData = tf.tensor(inputData);
        // Normalize or reshape data if necessary
        // Example: return tensorData.reshape([1, inputData.length]);
        return tensorData;
    }

    // Make predictions using the loaded model
    async predict(inputData) {
        if (!this.model) {
            throw new Error('Model is not loaded. Please load the model first.');
        }

        const processedData = this.preprocessData(inputData);
        const predictions = this.model.predict(processedData);
        const output = await predictions.array(); // Convert tensor to array
        return output;
    }

    // Example usage
    async exampleUsage() {
        // Load the model
        await this.loadModel();

        // Example input data for prediction
        const inputData = [1.0, 2.0, 3.0]; // Replace with actual input data

        // Make a prediction
        try {
            const prediction = await this.predict(inputData);
            console.log('Prediction:', prediction);
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    }
}

// Example usage
const modelPath = 'path/to/your/model.json'; // Replace with the actual path to your model
const mlAPI = new MachineLearningAPI(modelPath);
mlAPI.exampleUsage();

export default MachineLearningAPI;
