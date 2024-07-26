import { SidraChain } from '../sidra-chain';
import { TensorFlow } from 'tensorflow';

class AIBot {
  constructor(sidraChain) {
    this.sidraChain = sidraChain;
    this.tensorFlow = new TensorFlow();
  }

  async start() {
    // Load AI model
    const model = await this.tensorFlow.loadModel('ai-bot-model');
    // Start trading bot
    this.sidraChain.startTradingBot(model);
  }
}

export { AIBot };
