// tests/ai-auditor.test.js

const { AI Auditor } = require('../ai-auditor');
const { expect } = require('chai');

describe('AI Auditor', () => {
  let aiAuditor;

  beforeEach(() => {
    aiAuditor = new AI Auditor();
  });

  it('should audit AI models', async () => {
    const aiModel = {
      type: 'neural-network',
      layers: [
        { type: 'input', size: 784 },
        { type: 'hidden', size: 256 },
        { type: 'output', size: 10 }
      ]
    };
    const result = await aiAuditor.audit(aiModel);
    expect(result).to.have.property('accuracy');
    expect(result).to.have.property('bias');
    expect(result).to.have.property('explainability');
  });

  it('should throw an error for invalid AI models', async () => {
    const aiModel = {
      type: 'invalid',
      layers: []
    };
    try {
      await aiAuditor.audit(aiModel);
      throw new Error('Expected an error to be thrown');
    } catch (error) {
      expect(error).to.be.an.instanceof(Error);
    }
  });
});
