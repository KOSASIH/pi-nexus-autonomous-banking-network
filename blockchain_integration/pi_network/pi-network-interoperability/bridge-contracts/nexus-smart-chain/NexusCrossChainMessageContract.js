const Web3 = require('web3');
const Ethers = require('ethers');

class NexusCrossChainMessageContract {
  constructor(nexusCrossChainMessageContractAddress, providerUrl) {
    this.nexusCrossChainMessageContractAddress = nexusCrossChainMessageContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async sendMessage(messageId, message) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusCrossChainMessageContractAddress);
    const tx = {
      from: this.nexusCrossChainMessageContractAddress,
      to: this.nexusCrossChainMessageContractAddress,
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
      to: this.nexusCrossChainMessageContractAddress,
      data: `0x${messageId.toString(16)}`
    });
  }
}

module.exports = NexusCrossChainMessageContract;
