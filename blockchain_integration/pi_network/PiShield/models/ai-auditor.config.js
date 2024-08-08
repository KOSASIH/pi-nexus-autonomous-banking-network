// ai-auditor.config.js

module.exports = {
  // Neural network model configuration
  neuralNetwork: {
    inputShape: [100],
    hiddenLayers: [128, 64],
    outputShape: [1],
    optimizer: 'adam',
    lossFunction: 'meanSquaredError'
  },

  // Rule-based model configuration
  ruleBased: {
    rules: [
      {
        condition: 'transfer',
        score: 0.5
      },
      {
        condition: 'approve',
        score: 0.3
      },
      {
        condition: 'transferFrom',
        score: 0.2
      }
    ]
  },

  // AI auditor configuration
  aiAuditor: {
    models: ['neuralNetwork', 'ruleBased'],
    weights: [0.7, 0.3],
    threshold: 0.5
  }
};
