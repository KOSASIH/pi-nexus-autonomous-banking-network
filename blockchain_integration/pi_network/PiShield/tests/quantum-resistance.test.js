// tests/quantum-resistance.test.js

const { QuantumResistance } = require('../quantum-resistance');
const { expect } = require('chai');

describe('Quantum Resistance', () => {
  let quantumResistance;

  beforeEach(() => {
    quantumResistance = new QuantumResistance();
  });

  it('should generate a quantum-resistant key pair', async () => {
    const keyPair = await quantumResistance.generateKeyPair();
    expect(keyPair).to.have.property('publicKey');
    expect(keyPair).to.have.property('privateKey');
  });

  it('should encrypt and decrypt data using quantum-resistant cryptography', async () => {
    const plaintext = 'Hello, World!';
    const keyPair = await quantumResistance.generateKeyPair();
    const ciphertext = await quantumResistance.encrypt(plaintext, keyPair.publicKey);
    const decryptedText = await quantumResistance.decrypt(ciphertext, keyPair.privateKey);
    expect(decryptedText).to.equal(plaintext);
  });
});
