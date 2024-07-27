const Web3 = require('web3');
const Ethers = require('ethers');

class NexusBridgeContract {
  constructor(nexusBridgeContractAddress, providerUrl) {
    this.nexusBridgeContractAddress = nexusBridgeContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async bridgeToken(tokenAddress, recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusBridgeContractAddress);
    const tx = {
      from: this.nexusBridgeContractAddress,
      to: tokenAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getBalance(tokenAddress) {
    return this.web3.eth.getBalance(tokenAddress);
  }

  async getNexusBalance(address) {
    return this.web3.eth.getBalance(address);
  }
}

module.exports = NexusBridgeContract;
