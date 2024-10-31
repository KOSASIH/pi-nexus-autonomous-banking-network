// utils/validation.js

/**
 * Validate an Ethereum address
 * @param {string} address - The Ethereum address
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidAddress = (address) => {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
};

/**
 * Validate a number
 * @param {number} value - The value to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidNumber = (value) => {
    return typeof value === 'number' && !isNaN(value) && isFinite(value);
};

/**
 * Validate a string
 * @param {string} str - The string to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidString = (str) => {
    return typeof str === 'string' && str.trim().length > 0;
};

/**
 * Validate Ether amount
 * @param {string} ether - The Ether amount to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidEtherAmount = (ether) => {
    const amount = parseFloat(ether);
    return isValidNumber(amount) && amount > 0;
};

/**
 * Validate a transaction value
 * @param {string} value - The transaction value in Wei
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidTransactionValue = (value) => {
    const weiValue = BigInt(value);
    return weiValue > 0n;
};

/**
 * Validate a non-empty array
 * @param {Array} arr - The array to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidArray = (arr) => {
    return Array.isArray(arr) && arr.length > 0;
};

/**
 * Validate a function input
 * @param {Function} fn - The function to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidFunction = (fn) => {
    return typeof fn === 'function';
};

module.exports = {
    isValidAddress,
    isValidNumber,
    isValidString,
    isValidEtherAmount,
    isValidTransactionValue,
    isValidArray,
    isValidFunction,
};
