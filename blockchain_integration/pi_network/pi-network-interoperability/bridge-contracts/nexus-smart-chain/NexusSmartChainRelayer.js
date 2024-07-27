const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainRelayer {
  constructor(relayerContractAddress, providerUrl) {
    this.relayerContractAddress = relayerContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async relayTokens(amount, tokenAddress, userAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.relayerContractAddress);
    const tx = {
      from: this.relayerContractAddress,
      to: this.relayerContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeRelayTokens(amount, tokenAddress, userAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getRelayedTokens(tokenAddress, userAddress) {
    return this.api.query.nexusSmartChainRelayer.relayedTokens(tokenAddress, userAddress);
  }

    encodeRelayTokens(amount, tokenAddress, userAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'relayTokens',
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

module.exports = NexusSmartChainRelayer;
