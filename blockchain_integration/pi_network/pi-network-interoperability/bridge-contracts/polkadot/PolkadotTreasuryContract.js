const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotTreasuryContract {
  constructor(polkadotTreasuryContractAddress, providerUrl) {
    this.polkadotTreasuryContractAddress = polkadotTreasuryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async proposeSpend(proposalId, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotTreasuryContractAddress);
    const tx = {
      from: this.polkadotTreasuryContractAddress,
      to: this.polkadotTreasuryContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${proposalId.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async approveSpend(proposalId) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotTreasuryContractAddress);
    const tx = {
      from: this.polkadotTreasuryContractAddress,
      to: this.polkadotTreasuryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${proposalId.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getTreasuryBalance() {
    return this.api.query.treasury.balance();
  }
}

module.exports = PolkadotTreasuryContract;
