const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotStakingContract {
  constructor(polkadotStakingContractAddress, providerUrl) {
    this.polkadotStakingContractAddress = polkadotStakingContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async stake(amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotStakingContractAddress);
    const tx = {
      from: this.polkadotStakingContractAddress,
      to: this.polkadotStakingContractAddress,  
       value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async unstake(amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotStakingContractAddress);
    const tx = {
      from: this.polkadotStakingContractAddress,
      to: this.polkadotStakingContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getStakedBalance(address) {
    return this.api.query.staking.stakedBalance(address);
  }
}

module.exports = PolkadotStakingContract;
