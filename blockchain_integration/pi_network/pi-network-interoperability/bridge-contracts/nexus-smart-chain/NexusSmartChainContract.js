const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainContract {
  constructor(nexusSmartChainContractAddress, providerUrl) {
    this.nexusSmartChainContractAddress = nexusSmartChainContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async createSmartContract(code, data) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainContractAddress);
    const tx = {
      from: this.nexusSmartChainContractAddress,
      to: this.nexusSmartChainContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${code.toString(16)}${data.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async executeSmartContract(contractAddress, data) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusSmartChainContractAddress);
    const tx = {
      from: this.nexusSmartChainContractAddress,
      to: contractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${data.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getSmartContractCode(contractAddress) {
    return this.api.query.nexusSmartChain.contractCode(contractAddress);
  }

  async getSmartContractData(contractAddress) {
    return this.api.query.nexusSmartChain.contractData(contractAddress);
  }
}

module.exports = NexusSmartChainContract;
