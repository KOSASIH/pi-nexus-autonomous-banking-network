const crypto = require('crypto');
const elliptic = require('elliptic');

const ec = elliptic.ec('secp256k1');

const CryptoUtils = {
  /**
   * Generate a random 32-byte private key
   */
  generatePrivateKey: () => {
    return crypto.randomBytes(32).toString('hex');
  },

  /**
   * Derive a public key from a private key
   */
  derivePublicKey: (privateKey) => {
    const keyPair = ec.keyFromPrivate(privateKey, 'hex');
    return keyPair.getPublic().encode('hex', true);
  },

  /**
   * Sign a message with a private key
   */
  signMessage: (privateKey, message) => {
    const keyPair = ec.keyFromPrivate(privateKey, 'hex');
    const signature = keyPair.sign(message);
    return signature.toDER('hex');
  },

  /**
   * Verify a signature with a public key
   */
  verifySignature: (publicKey, message, signature) => {
    const keyPair = ec.keyFromPublic(publicKey, 'hex');
    return keyPair.verify(message, signature);
  },

  /**
   * Hash a message using SHA-256
   */
  hashMessage: (message) => {
    return crypto.createHash('sha256').update(message).digest('hex');
  },
};

module.exports = CryptoUtils;
