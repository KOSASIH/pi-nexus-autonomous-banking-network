import { EdgeAI } from 'edge-ai-sdk';

class EdgeAI {
  constructor() {
    this.edgeAI = new EdgeAI();
  }

  async processAIModelAtEdge(model, data) {
    const result = await this.edgeAI.process(model, data);
    return result;
  }
}

export default EdgeAI;
