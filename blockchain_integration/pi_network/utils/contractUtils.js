// utils/contractUtils.js
const Web3 = require('web3');
const web3 = new Web3(Web3.givenProvider || "http://localhost:8545");

/**
 * Create a contract instance
 * @param {string} abi - The contract ABI
 * @param {string} address - The contract address
 * @returns {Object} - The contract instance
 */
const getContractInstance = (abi, address) => {
    return new web3.eth.Contract(abi, address);
};

/**
 * Send a transaction to a contract method
 * @param {Object} contract - The contract instance
 * @param {string} methodName - The method name to call
 * @param {Array} args - The arguments for the method
 * @param {string} from - The address sending the transaction
 * @param {string} value - The value to send (in Wei)
 * @returns {Promise<Object>} - The transaction receipt
 */
const sendTransaction = async (contract, methodName, args, from, value = '0') => {
    const method = contract.methods[methodName](...args);
    const gas = await method.estimateGas({ from, value });
    return await method.send({ from, gas, value });
};

/**
 * Call a contract method (read-only)
 * @param {Object} contract - The contract instance
 * @param {string} methodName - The method name to call
 * @param {Array} args - The arguments for the method
 * @returns {Promise<any>} - The result of the method call
 */
const callMethod = async (contract, methodName, args) => {
    return await contract.methods[methodName](...args).call();
};

module.exports = {
    getContractInstance,
    sendTransaction,
    callMethod,
};
