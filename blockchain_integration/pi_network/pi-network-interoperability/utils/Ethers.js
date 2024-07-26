const Ethers = require('ethers');

class EthersUtils {
  constructor(providerUrl) {
    this.ethers = new Ethers(new Ethers.providers.JsonRpcProvider(providerUrl));
  }

  async getBlockNumber() {
    return this.ethers.getBlockNumber();
  }

  async getTransactionCount(address) {
    return this.ethers.getTransactionCount(address);
  }

  async getBalance(address) {
    return this.ethers.getBalance(address);
  }

  async sendTransaction(tx) {
    return this.ethers.sendTransaction(tx);
  }

  async getTransactionReceipt(txHash) {
    return this.ethers.getTransactionReceipt(txHash);
  }
}

module.exports = EthersUtils;
