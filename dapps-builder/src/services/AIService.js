import { AIEngine } from '../components/AIEngine';
import { preprocessData } from '../components/AIEngine/utils';

class AIService {
    constructor() {
        this.model = null;
    }

    async train(data) {
        const processedData = preprocessData(data);
        this.model = await AIEngine.train(processedData);
        return this.model;
    }

    async loadModel(modelPath) {
        this.model = await AIEngine.load(modelPath);
        return this.model;
    }

    async predict(input) {
        if (!this.model) {
            throw new Error('Model not loaded or trained.');
        }
        return await AIEngine.predict(this.model, input);
    }

    async saveModel(path) {
        if (!this.model) {
            throw new Error('Model not trained or loaded.');
        }
        await AIEngine.saveModel(this.model, path);
    }
}

export default new AIService();
