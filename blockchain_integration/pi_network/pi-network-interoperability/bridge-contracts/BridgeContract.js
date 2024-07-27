const Web3 = require('web3');
const Ethers = require('ethers');

class BridgeContract {
  constructor(bridgeContractAddress, providerUrl) {
    this.bridgeContractAddress = bridgeContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async bridgeToken(tokenAddress, recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.bridgeContractAddress);
    const tx = {
      from: this.bridgeContractAddress,
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
}

module.exports = BridgeContract;
