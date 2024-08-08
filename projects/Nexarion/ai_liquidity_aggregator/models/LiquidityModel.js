import { LSTM } from 'machinelearn';

class LiquidityModel {
  constructor() {
    this.model = new LSTM({
      inputShape: [10, 10], // 10 timesteps, 10 features
      outputShape: [1], // 1 output feature
      layers: [
        { type: 'lstm', units: 50, returnSequences: true },
        { type: 'dense', units: 1 }
      ]
    });
  }

  train(data) {
    this.model.fit(data, { epochs: 100 });
  }

  predict(data) {
    return this.model.predict(data);
  }
}

export default LiquidityModel;
