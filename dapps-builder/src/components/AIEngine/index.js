import { trainModel, loadModel } from './modelTraining';
import { makePrediction } from './inference';

export const AIEngine = {
    train: async (data) => {
        return await trainModel(data);
    },
    load: async (modelPath) => {
        return await loadModel(modelPath);
    },
    predict: async (input) => {
        return await makePrediction(input);
    }
};
