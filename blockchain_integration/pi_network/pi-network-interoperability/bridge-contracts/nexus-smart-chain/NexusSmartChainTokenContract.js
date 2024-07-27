const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainTokenContract {
  constructor(nexusSmartChainTokenContractAddress, providerUrl) {
    this.nexusSmartChainTokenContractAddress = nexusSmartChainTokenContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async mintTokens(amount, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainTokenContractAddress);
    const tx = {
      from: this.nexusSmartChainTokenContractAddress,
      to: this.nexusSmartChainTokenContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeMintTokens(amount, toAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async burnTokens(amount, fromAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainTokenContractAddress);
    const tx = {
      from: this.nexusSmartChainTokenContractAddress,
      to: this.nexusSmartChainTokenContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeBurnTokens(amount, fromAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async transferTokens(amount, fromAddress, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainTokenContractAddress);
    const tx = {
      from: this.nexusSmartChainTokenContractAddress,
      to: this.nexusSmartChainTokenContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeTransferTokens(amount, fromAddress, toAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getBalance(address) {
    return this.api.query.nexusSmartChainToken.balanceOf(address);
  }

  async getTotalSupply() {
    return this.api.query.nexusSmartChainToken.totalSupply();
  }

  encodeMintTokens(amount, toAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'mintTokens',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'toAddress'
      }]
    }, [amount, toAddress]);
  }

  encodeBurnTokens(amount, fromAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'burnTokens',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'fromAddress'
      }]
    }, [amount, fromAddress]);
  }

  encodeTransferTokens(amount, fromAddress, toAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'transferTokens',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'fromAddress'
      }, {
        type: 'address',
        name: 'toAddress'
      }]
    }, [amount, fromAddress, toAddress]);
  }
}

module.exports = NexusSmartChainTokenContract;
