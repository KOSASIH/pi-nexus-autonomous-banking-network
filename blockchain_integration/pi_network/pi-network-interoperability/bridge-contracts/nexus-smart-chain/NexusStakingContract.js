const Web3 = require('web3');
const Ethers = require('ethers');

class NexusStakingContract {
  constructor(nexusStakingContractAddress, providerUrl) {
    this.nexusStakingContractAddress = nexusStakingContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async stake(amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusStakingContractAddress);
    const tx = {
      from: this.nexusStakingContractAddress,
      to: this.nexusStakingContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async unstake(amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusStakingContractAddress);
    const tx = {
      from: this.nexusStakingContractAddress,
      to: this.nexusStakingContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getStakedBalance(address) {
    return this.web3.eth.call({
      to: this.nexusStakingContractAddress,
      data: `0x70a08231000000000000000000000000${address.slice(2)}`
    });
  }
}

module.exports = NexusStakingContract;
