// utils/web3Utils.js
const Web3 = require('web3');

const web3 = new Web3(Web3.givenProvider || "http://localhost:8545");

/**
 * Convert Ether to Wei
 * @param {string} ether - The amount in Ether
 * @returns {string} - The amount in Wei
 */
const toWei = (ether) => {
    return web3.utils.toWei(ether, 'ether');
};

/**
 * Convert Wei to Ether
 * @param {string} wei - The amount in Wei
 * @returns {string} - The amount in Ether
 */
const fromWei = (wei) => {
    return web3.utils.fromWei(wei, 'ether');
};

/**
 * Get the current network ID
 * @returns {Promise<number>} - The current network ID
 */
const getNetworkId = async () => {
    return await web3.eth.net.getId();
};

/**
 * Get the balance of an address
 * @param {string} address - The Ethereum address
 * @returns {Promise<string>} - The balance in Wei
 */
const getBalance = async (address) => {
    return await web3.eth.getBalance(address);
};

module.exports = {
    toWei,
    fromWei,
    getNetworkId,
    getBalance,
};
