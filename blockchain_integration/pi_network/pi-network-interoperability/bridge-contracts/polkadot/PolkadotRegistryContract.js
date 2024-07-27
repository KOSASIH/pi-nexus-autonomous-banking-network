const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotRegistryContract {
  constructor(polkadotRegistryContractAddress, providerUrl) {
    this.polkadotRegistryContractAddress = polkadotRegistryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async registerNode(nodeId, nodeAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotRegistryContractAddress);
    const tx = {
      from: this.polkadotRegistryContractAddress,
      to: this.polkadotRegistryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${nodeId.toString(16)}${nodeAddress}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getNode(nodeId) {
    return this.api.query.registry.nodeOf(nodeId);
  }
}

module.exports = PolkadotRegistryContract;
