const Web3 = require('web3');
const Ethers = require('ethers');

class CrossChainMessageContract {
  constructor(crossChainMessageContractAddress, providerUrl) {
    this.crossChainMessageContractAddress = crossChainMessageContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async sendMessage(messageId, message) {
    const txCount = await this.web3.eth.getTransactionCount(this.crossChainMessageContractAddress);
    const tx = {
      from: this.crossChainMessageContractAddress,
      to: this.crossChainMessageContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${messageId.toString(16)}${message}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getMessage(messageId) {
    return this.web3.eth.call({
      to: this.crossChainMessageContractAddress,
      data: `0x${messageId.toString(16)}`
    });
  }
}

module.exports = CrossChainMessageContract;
