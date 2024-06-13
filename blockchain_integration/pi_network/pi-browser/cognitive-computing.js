import { CognitiveComputing } from 'cognitive-computing-sdk';

class CognitiveComputing {
  constructor() {
    this.cognitiveComputing = new CognitiveComputing();
  }

  async analyzeUnstructuredData(data) {
    const insights = await this.cognitiveComputing.analyze(data);
    return insights;
  }
}

export default CognitiveComputing;
