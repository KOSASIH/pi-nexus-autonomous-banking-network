import { NLU } from 'nlu-sdk';

class NaturalLanguageUnderstanding {
  constructor() {
    this.nlu = newNLU();
  }

  async analyzeNaturalLanguageText(text) {
    const insights = await this.nlu.analyze(text);
    return insights;
  }
}

export default NaturalLanguageUnderstanding;
