// crypto.config.js

module.exports = {
  // Lattice-based cryptography configuration
  lattice: {
    securityParameter: 2048,
    ntruParameterSet: 'ntru-hrss-2048-761'
  },

  // Hash-based signatures configuration
  hashBasedSignatures: {
    hashAlgorithm: 'sha256',
    signatureAlgorithm: 'rsa-sha256',
    privateKey: 'path/to/private/key',
    publicKey: 'path/to/public/key'
  }
};
