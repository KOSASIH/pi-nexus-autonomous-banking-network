import { BrainJS } from 'brain.js';
import { NaturalLanguageUnderstandingV1 } from 'ibm-watson';

class AIAssistant {
  constructor() {
    this.brain = new BrainJS.NeuralNetwork();
    this.nlu = new NaturalLanguageUnderstandingV1({
      iam_apikey: 'YOUR_API_KEY',
      version: '2021-08-01',
    });
  }

  async processUserInput(input) {
    const intent = await this.nlu.analyze({
      text: input,
      features: {
        sentiment: {},
        entities: {},
        keywords: {},
      },
    });

    const response = this.brain.run(intent.sentiment.document.label);
    return response;
  }
}

export default AIAssistant;
