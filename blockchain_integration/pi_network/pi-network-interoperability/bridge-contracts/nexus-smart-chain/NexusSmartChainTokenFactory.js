const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainTokenFactory {
  constructor(factoryContractAddress, providerUrl) {
    this.factoryContractAddress = factoryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async createToken(name, symbol, decimals, totalSupply) {
    const txCount = await this.web3.eth.getTransactionCount(this.factoryContractAddress);
    const tx = {
      from: this.factoryContractAddress,
      to: this.factoryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
            nonce: txCount,
      data: `0x${this.encodeCreateToken(name, symbol, decimals, totalSupply)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getTokenAddress(name, symbol) {
    return this.api.query.nexusSmartChainTokenFactory.tokenAddress(name, symbol);
  }

  encodeCreateToken(name, symbol, decimals, totalSupply) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'createToken',
      type: 'function',
      inputs: [{
        type: 'string',
        name: 'name'
      }, {
        type: 'string',
        name: 'symbol'
      }, {
        type: 'uint8',
        name: 'decimals'
      }, {
        type: 'uint256',
        name: 'totalSupply'
      }]
    }, [name, symbol, decimals, totalSupply]);
  }
}

module.exports = NexusSmartChainTokenFactory;
