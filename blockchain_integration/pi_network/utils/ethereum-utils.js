const Web3 = require('web3');
const ethUtils = require('ethereumjs-util');
const utils = {
  toDecimal: (hex) => parseInt(hex, 16),
  toHex: (decimal) => decimal.toString(16),
  calculateGasFee: (gasPrice, gasUsed) => gasPrice * gasUsed,
  toChecksumAddress: (address) => ethUtils.toChecksumAddress(address),
  isAddress: (address) => ethUtils.isValidAddress(address),
  toWei: (value, unit) => web3.utils.toWei(value, unit),
  fromWei: (value, unit) => web3.utils.fromWei(value, unit),
};

const web3 = new Web3();
module.exports = utils;
