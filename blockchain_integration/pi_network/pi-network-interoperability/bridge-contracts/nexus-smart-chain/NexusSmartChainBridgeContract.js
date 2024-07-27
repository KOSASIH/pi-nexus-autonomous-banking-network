const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainBridgeContract {
  constructor(nexusSmartChainBridgeContractAddress, providerUrl) {
    this.nexusSmartChainBridgeContractAddress = nexusSmartChainBridgeContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async lockTokens(amount, tokenAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainBridgeContractAddress);
    const tx = {
      from: this.nexusSmartChainBridgeContractAddress,
      to: this.nexusSmartChainBridgeContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${tokenAddress.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async unlockTokens(amount, tokenAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainBridgeContractAddress);
    const tx = {
      from: this.nexusSmartChainBridgeContractAddress,
      to: this.nexusSmartChainBridgeContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${tokenAddress.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getLockedTokens(tokenAddress) {
    return this.api.query.nexusSmartChainBridge.lockedTokens(tokenAddress);
  }
}

module.exports = NexusSmartChainBridgeContract;
