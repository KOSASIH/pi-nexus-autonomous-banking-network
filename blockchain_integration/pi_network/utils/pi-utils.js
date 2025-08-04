const utils = {
  toDecimal: (hex) => parseInt(hex, 16),
  toHex: (decimal) => decimal.toString(16),
  calculateGasFee: (gasPrice, gasUsed) => gasPrice * gasUsed,
};

module.exports = utils;
