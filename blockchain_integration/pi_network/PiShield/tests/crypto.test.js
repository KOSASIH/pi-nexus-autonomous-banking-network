// tests/crypto.test.js

const { Crypto } = require('../crypto');
const { expect } = require('chai');

describe('Crypto', () => {
  let crypto;

  beforeEach(() => {
    crypto = new Crypto();
  });

  it('should encrypt and decrypt data', async () => {
    const plaintext = 'Hello, World!';
    const ciphertext = await crypto.encrypt(plaintext);
    const decryptedText = await crypto.decrypt(ciphertext);
    expect(decryptedText).to.equal(plaintext);
  });

  it('should throw an error for invalid encryption keys', async () => {
    try {
      await crypto.encrypt('Hello, World!', 'invalid-key');
      throw new Error('Expected an error to be thrown');
    } catch (error) {
      expect(error).to.be.an.instanceof(Error);
    }
  });
});
