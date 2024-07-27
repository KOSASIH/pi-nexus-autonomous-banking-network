const Web3 = require('web3');
const Ethers = require('ethers');

class NexusTokenContract {
  constructor(nexusTokenContractAddress, providerUrl) {
    this.nexusTokenContractAddress = nexusTokenContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async transfer(recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusTokenContractAddress);
    const tx = {
      from: this.nexusTokenContractAddress,
      to: recipientAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async approve(spenderAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusTokenContractAddress);
    const tx = {
      from: this.nexusTokenContractAddress,
      to: spenderAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async balanceOf(address) {
    return this.web3.eth.call({
      to: this.nexusTokenContractAddress,
      data: `0x70a08231000000000000000000000000${address.slice(2)}`
    });
  }
}

module.exports = NexusTokenContract;
