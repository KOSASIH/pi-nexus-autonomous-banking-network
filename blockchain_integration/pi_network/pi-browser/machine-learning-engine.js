import { MachineLearningEngine } from 'achine-learning-engine-sdk';

class MachineLearningEngine {
  constructor() {
    this.machineLearningEngine = new MachineLearningEngine();
  }

  async trainMachineLearningModel(model, data) {
    const trainedModel = await this.machineLearningEngine.train(model, data);
    return trainedModel;
  }
}

export default MachineLearningEngine;
