const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainRelayerContract {
  constructor(nexusSmartChainRelayerContractAddress, providerUrl) {
    this.nexusSmartChainRelayerContractAddress = nexusSmartChainRelayerContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async relayTokens(amount, tokenAddress, recipient) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainRelayerContractAddress);
    const tx = {
      from: this.nexusSmartChainRelayerContractAddress,
      to: this.nexusSmartChainRelayerContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${tokenAddress.toString(16)}${recipient.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getRelayedTokens(tokenAddress) {
    return this.api.query.nexusSmartChainRelayer.relayedTokens(tokenAddress);
  }
}

module.exports = NexusSmartChainRelayerContract;
