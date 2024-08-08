// tests/formal-verifier.test.js

const { FormalVerifier } = require('../formal-verifier');
const { expect } = require('chai');

describe('Formal Verifier', () => {
  let formalVerifier;

  beforeEach(() => {
    formalVerifier = new FormalVerifier();
  });

  it('should verify formal proofs', async () => {
    const proof = {
      statement: 'forall x, P(x)',
      proofSteps: [
        { type: 'axiom', statement: 'P(0)' },
        { type: 'modus-ponens', premises: ['P(0)', 'P(0) => P(1)'], conclusion: 'P(1)' },
        { type: 'induction', premise: 'P(1)', conclusion: 'forall x, P(x)' }
      ]
    };
    const result = await formalVerifier.verify(proof);
    expect(result).to.be.true;
  });

  it('should reject invalid formal proofs', async () => {
    const proof = {
      statement: 'forall x, P(x)',
      proofSteps: [
        { type: 'axiom', statement: 'P(0)' },
        { type: 'modus-ponens', premises: ['P(0)', 'P(0) => Q(1)'], conclusion: 'P(1)' },
        { type: 'induction', premise: 'P(1)', conclusion: 'forall x, P(x)' }
      ]
    };
    try {
      await formalVerifier.verify(proof);
      throw new Error('Expected an error to be thrown');
    } catch (error) {
      expect(error).to.be.an.instanceof(Error);
    }
  });
});
