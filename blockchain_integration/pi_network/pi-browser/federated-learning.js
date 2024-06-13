import { FederatedLearning } from 'federated-learning-sdk';

class FederatedLearning {
  constructor() {
    this.federatedLearning = new FederatedLearning();
  }

  async trainModel(model, data) {
    const trainedModel = await this.federatedLearning.train(model, data);
    return trainedModel;
  }
}

export default FederatedLearning;
