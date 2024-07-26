const Web3 = require('web3');

class Web3Utils {
  constructor(providerUrl) {
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
  }

  async getBlockNumber() {
    return this.web3.eth.getBlockNumber();
  }

  async getTransactionCount(address) {
    return this.web3.eth.getTransactionCount(address);
  }

  async getBalance(address) {
    return this.web3.eth.getBalance(address);
  }

  async sendTransaction(tx) {
    return this.web3.eth.sendTransaction(tx);
  }

  async getTransactionReceipt(txHash) {
    return this.web3.eth.getTransactionReceipt(txHash);
  }
}

module.exports = Web3Utils;
