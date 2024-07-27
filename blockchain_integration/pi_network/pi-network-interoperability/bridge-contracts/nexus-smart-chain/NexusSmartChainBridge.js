const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainBridge {
  constructor(bridgeContractAddress, providerUrl) {
    this.bridgeContractAddress = bridgeContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async lockTokens(amount, tokenAddress, userAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.bridgeContractAddress);
    const tx = {
      from: this.bridgeContractAddress,
      to: this.bridgeContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeLockTokens(amount, tokenAddress, userAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async unlockTokens(amount, tokenAddress, userAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.bridgeContractAddress);
    const tx = {
      from: this.bridgeContractAddress,
      to: this.bridgeContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeUnlockTokens(amount, tokenAddress, userAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getLockedTokens(tokenAddress, userAddress) {
    return this.api.query.nexusSmartChainBridge.lockedTokens(tokenAddress, userAddress);
  }

  encodeLockTokens(amount, tokenAddress, userAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'lockTokens',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'tokenAddress'
      }, {
        type: 'address',
        name: 'userAddress'
      }]
    }, [amount, tokenAddress, userAddress]);
  }

  encodeUnlockTokens(amount, tokenAddress, userAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'unlockTokens',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'tokenAddress'
      }, {
        type: 'address',
        name: 'userAddress'
      }]
    }, [amount, tokenAddress, userAddress]);
  }
}

module.exports = NexusSmartChainBridge;
