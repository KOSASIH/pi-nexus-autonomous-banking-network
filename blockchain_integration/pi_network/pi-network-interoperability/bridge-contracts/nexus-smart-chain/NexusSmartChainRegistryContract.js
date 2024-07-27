const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainRegistryContract {
  constructor(nexusSmartChainRegistryContractAddress, providerUrl) {
    this.nexusSmartChainRegistryContractAddress = nexusSmartChainRegistryContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async registerSmartContract(contractAddress, code, data) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainRegistryContractAddress);
    const tx = {
      from: this.nexusSmartChainRegistryContractAddress,
      to: this.nexusSmartChainRegistryContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${contractAddress.toString(16)}${code.toString(16)}${data.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getRegisteredSmartContracts() {
    return this.api.query.nexusSmartChainRegistry.registeredContracts();
  }

  async getSmartContractInfo(contractAddress) {
    return this.api.query.nexusSmartChainRegistry.contractInfo(contractAddress);
  }
}

module.exports = NexusSmartChainRegistryContract;
