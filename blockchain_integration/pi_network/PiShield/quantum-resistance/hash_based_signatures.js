// hash_based_signatures.js

const crypto = require('crypto');

class HashBasedSignatures {
  constructor(params) {
    this.params = params;
    this.hashFunction = crypto.createHash(params.hashAlgorithm);
  }

  sign(message, privateKey) {
    // Sign the message using the private key
    const signature = crypto.createHmac(this.params.hashAlgorithm, privateKey).update(message).digest();
    return signature;
  }

  verify(message, signature, publicKey) {
    // Verify the signature using the public key
    const expectedSignature = crypto.createHmac(this.params.hashAlgorithm, publicKey).update(message).digest();
    return crypto.timingSafeEqual(signature, expectedSignature);
  }
}

module.exports = HashBasedSignatures;
