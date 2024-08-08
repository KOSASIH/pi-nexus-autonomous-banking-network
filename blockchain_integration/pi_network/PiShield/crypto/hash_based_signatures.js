// hash_based_signatures.js

const crypto = require('crypto');

class HashBasedSignatures {
  constructor(config) {
    this.config = config;
  }

  async sign(message, privateKey) {
    const hash = crypto.createHash(this.config.hashAlgorithm);
    hash.update(message);
    const hashedMessage = hash.digest();
    const signature = crypto.sign(this.config.signatureAlgorithm, hashedMessage, privateKey);
    return signature;
  }

  async verify(message, signature, publicKey) {
    const hash = crypto.createHash(this.config.hashAlgorithm);
    hash.update(message);
    const hashedMessage = hash.digest();
    const isValid = crypto.verify(this.config.signatureAlgorithm, hashedMessage, publicKey, signature);
    return isValid;
  }
}

module.exports = HashBasedSignatures;
