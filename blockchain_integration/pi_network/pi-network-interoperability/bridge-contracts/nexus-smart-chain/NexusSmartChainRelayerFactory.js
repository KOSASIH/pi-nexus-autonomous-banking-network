const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainRelayerFactory {
  constructor(relayerFactoryContractAddress, providerUrl) {
    this.relayerFactoryContractAddress = relayerFactoryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async createRelayer(tokenAddress, relayerOwner) {
    const txCount = await this.web3.eth.getTransactionCount(this.relayerFactoryContractAddress);
    const tx = {
      from: this.relayerFactoryContractAddress,
      to: this.relayerFactoryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeCreateRelayer(tokenAddress, relayerOwner)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getRelayer(tokenAddress) {
    return this.api.query.nexusSmartChainRelayerFactory.relayers(tokenAddress);
  }

  encodeCreateRelayer(tokenAddress, relayerOwner) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'createRelayer',
      type: 'function',
      inputs: [{
        type: 'address',
        name: 'tokenAddress'
      }, {
        type: 'address',
        name: 'elayerOwner'
      }]
    }, [tokenAddress, relayerOwner]);
  }
}

module.exports = NexusSmartChainRelayerFactory;
