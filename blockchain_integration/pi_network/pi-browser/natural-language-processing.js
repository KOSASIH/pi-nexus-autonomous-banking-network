import { spaCy } from 'spacy';

class NaturalLanguageProcessing {
  constructor() {
    this.spacy = new spaCy();
  }

  async processText(text) {
    const doc = await this.spacy.process(text);
    return doc;
  }
}

export default NaturalLanguageProcessing;
