/**
 * Converts a decimal number to a hexadecimal string.
 *
 * @param {number} decimal - The decimal number to convert.
 * @returns {string} The hexadecimal string representation of the decimal number.
 */
function decimalToHex(decimal) {
  return decimal.toString(16);
}

/**
 * Converts a hexadecimal string to a decimal number.
 *
 * @param {string} hex - The hexadecimal string to convert.
 * @returns {number} The decimal number representation of the hexadecimal string.
 */
function hexToDecimal(hex) {
  return parseInt(hex, 16);
}

/**
 * Converts a decimal block number to a hexadecimal string.
 *
 * @param {number} blockNumber - The decimal block number to convert.
 * @returns {string} The hexadecimal string representation of the block number.
 */
function decimalBlockNumberToHex(blockNumber) {
  return decimalToHex(blockNumber).padStart(64, '0');
}

/**
 * Converts a hexadecimal block number to a decimal number.
 *
 * @param {string} hexBlockNumber - The hexadecimal block number to convert.
 * @returns {number} The decimal number representation of the block number.
 */
function hexBlockNumberToDecimal(hexBlockNumber) {
  return hexToDecimal(hexBlockNumber);
}

/**
 * Converts a transaction hash to a hexadecimal string.
 *
 * @param {string} transactionHash - The transaction hash to convert.
 * @returns {string} The hexadecimal string representation of the transaction hash.
 */
function transactionHashToHex(transactionHash) {
  return transactionHash.padStart(64, '0');
}

/**
 * Exports the conversion functions.
 */
module.exports = {
  decimalToHex,
  hexToDecimal,
  decimalBlockNumberToHex,
  hexBlockNumberToDecimal,
  transactionHashToHex,
};
