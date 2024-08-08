// quantum-resistance.config.js

module.exports = {
  // Lattice-based cryptography parameters
  lattice: {
    dimension: 256,
    byteSize: 32
  },

  // Hash-based signature parameters
  hash: {
    algorithm: 'sha3-256',
    saltSize: 16
  },

  // Quantum resistance parameters
  quantum: {
    securityLevel: 128,
    errorCorrection: true
  }
};
