import { AGI } from 'agi-sdk';

class ArtificialGeneralIntelligence {
  constructor() {
    this.agi = new AGI();
  }

  async processComplexTask(task) {
    const result = await this.agi.process(task);
    return result;
  }
}

export default ArtificialGeneralIntelligence;
