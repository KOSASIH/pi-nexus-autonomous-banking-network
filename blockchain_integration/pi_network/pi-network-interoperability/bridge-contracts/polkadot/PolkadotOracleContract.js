const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotOracleContract {
  constructor(polkadotOracleContractAddress, providerUrl) {
    this.polkadotOracleContractAddress = polkadotOracleContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async requestPriceFeed(symbol) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotOracleContractAddress);
    const tx = {
      from: this.polkadotOracleContractAddress,
      to: this.polkadotOracleContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${symbol}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getPriceFeed(symbol) {
    return this.api.query.oracle.getPriceFeed(symbol);
  }
}

module.exports = PolkadotOracleContract;
