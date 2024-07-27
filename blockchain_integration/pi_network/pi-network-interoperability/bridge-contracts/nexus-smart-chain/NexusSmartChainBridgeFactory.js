const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainBridgeFactory {
  constructor(bridgeFactoryContractAddress, providerUrl) {
    this.bridgeFactoryContractAddress = bridgeFactoryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async createBridge(tokenAddress, bridgeOwner) {
    const txCount = await this.web3.eth.getTransactionCount(this.bridgeFactoryContractAddress);
    const tx = {
      from: this.bridgeFactoryContractAddress,
      to: this.bridgeFactoryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeCreateBridge(tokenAddress, bridgeOwner)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getBridge(tokenAddress) {
    return this.api.query.nexusSmartChainBridgeFactory.bridges(tokenAddress);
  }

  encodeCreateBridge(tokenAddress, bridgeOwner) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'createBridge',
      type: 'function',
      inputs: [{
        type: 'address',
        name: 'tokenAddress'
      }, {
        type: 'address',
        name: 'bridgeOwner'
      }]
    }, [tokenAddress, bridgeOwner]);
  }
}

module.exports = NexusSmartChainBridgeFactory;
