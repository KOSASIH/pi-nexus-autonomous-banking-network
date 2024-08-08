// lattice_crypto.js

const crypto = require('crypto');

class LatticeCrypto {
  constructor(config) {
    this.config = config;
    this.ntru = require('ntrujs');
  }

  async generateKeyPair() {
    const keyPair = await this.ntru.generateKeyPair(this.config.securityParameter);
    return {
      publicKey: keyPair.publicKey,
      privateKey: keyPair.privateKey
    };
  }

  async encrypt(plaintext, publicKey) {
    const ciphertext = await this.ntru.encrypt(plaintext, publicKey);
    return ciphertext;
  }

  async decrypt(ciphertext, privateKey) {
    const plaintext = await this.ntru.decrypt(ciphertext, privateKey);
    return plaintext;
  }

  async sign(message, privateKey) {
    const signature = await this.ntru.sign(message, privateKey);
    return signature;
  }

  async verify(message, signature, publicKey) {
    const isValid = await this.ntru.verify(message, signature, publicKey);
    return isValid;
  }
}

module.exports = LatticeCrypto;
