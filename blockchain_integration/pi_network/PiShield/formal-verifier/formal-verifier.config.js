// formal-verifier.config.js

module.exports = {
  // Model checker configuration
  modelChecker: {
    timeout: 10000,
    maxIterations: 1000
  },

  // Theorem prover configuration
  theoremProver: {
    timeout: 10000,
    maxIterations: 1000
  },

  // Z3 configuration
  z3: {
    path: 'path/to/z3/bin',
    options: ['-smt2', '-in']
  }
};
