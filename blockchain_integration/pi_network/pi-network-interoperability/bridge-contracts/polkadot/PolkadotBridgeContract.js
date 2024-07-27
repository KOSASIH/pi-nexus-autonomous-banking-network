const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotBridgeContract {
  constructor(polkadotBridgeContractAddress, providerUrl) {
    this.polkadotBridgeContractAddress = polkadotBridgeContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async bridgeToken(tokenAddress, recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotBridgeContractAddress);
    const tx = {
      from: this.polkadotBridgeContractAddress,
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

  async getPolkadotBalance(address) {
    return this.api.query.balances.freeBalance(address);
  }
}

module.exports = PolkadotBridgeContract;
