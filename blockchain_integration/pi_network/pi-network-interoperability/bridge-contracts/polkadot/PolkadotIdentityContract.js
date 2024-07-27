const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotIdentityContract {
  constructor(polkadotIdentityContractAddress, providerUrl) {
    this.polkadotIdentityContractAddress = polkadotIdentityContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async createIdentity(identityId, publicKey) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotIdentityContractAddress);
    const tx = {
      from: this.polkadotIdentityContractAddress,
      to: this.polkadotIdentityContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${identityId.toString(16)}${publicKey}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async updateIdentity(identityId, publicKey) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotIdentityContractAddress);
    const tx = {
      from: this.polkadotIdentityContractAddress,
      to: this.polkadotIdentityContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${identityId.toString(16)}${publicKey}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getIdentity(identityId) {
    return this.api.query.identity.identityOf(identityId);
  }
}

module.exports = PolkadotIdentityContract;
