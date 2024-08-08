const Web3 = require('web3');

const Web3Utils = {
  /**
   * Convert a Web3.js transaction receipt to a human-readable format
   */
  formatTransactionReceipt: (receipt) => {
    return {
      blockNumber: receipt.blockNumber,
      transactionHash: receipt.transactionHash,
      gasUsed: receipt.gasUsed,
      status: receipt.status,
    };
  },

  /**
   * Convert a Web3.js block to a human-readable format
   */
  formatBlock: (block) => {
    return {
      number: block.number,
      hash: block.hash,
      timestamp: block.timestamp,
      transactions: block.transactions.map((tx) => tx.hash),
    };
  },

  /**
   * Get the current block number
   */
  getCurrentBlockNumber: async (web3) => {
    return web3.eth.getBlockNumber();
  },

  /**
   * Get the current gas price
   */
  getCurrentGasPrice: async (web3) => {
    return web3.eth.getGasPrice();
  },

  /**
   * Estimate the gas required for a transaction
   */
  estimateGas: async (web3, tx) => {
    return web3.eth.estimateGas(tx);
  },
};

module.exports = Web3Utils;
