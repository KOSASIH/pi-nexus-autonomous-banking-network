const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotTokenContract {
  constructor(polkadotTokenContractAddress, providerUrl) {
    this.polkadotTokenContractAddress = polkadotTokenContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async transfer(recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotTokenContractAddress);
    const tx = {
      from: this.polkadotTokenContractAddress,
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
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotTokenContractAddress);
    const tx = {
      from: this.polkadotTokenContractAddress,
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
    return this.api.query.tokens.balanceOf(address);
  }
}

module.exports = PolkadotTokenContract;
