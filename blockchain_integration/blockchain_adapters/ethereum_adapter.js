const Web3 = require('web3');
const { ChainId, ChainName } = require('../constants');

class EthereumAdapter {
  constructor() {
    this.web3 = new Web3(process.env.ETHEREUM_NODE_URL);
    this.chainId = ChainId.ETHEREUM;
    this.chainName = ChainName.ETHEREUM;
  }

  async getBalance(address) {
    const balance = await this.web3.eth.getBalance(address);
    return balance;
  }

  async sendTransaction(fromAddress, toAddress, value) {
    const gasPrice = await this.web3.eth.getGasPrice();
    const gas = await this.web3.eth.estimateGas({ from: fromAddress, to: toAddress, value });
    const transaction = {
      from: fromAddress,
      to: toAddress,
      value,
      gasPrice,
      gas,
    };
    const signedTransaction = await this.web3.eth.accounts.signTransaction(transaction, process.env.ETHEREUM_PRIVATE_KEY);
    const receipt = await this.web3.eth.sendSignedTransaction(signedTransaction.rawTransaction);
    return receipt;
  }

  async getTransactionReceipt(transactionHash) {
    const receipt = await this.web3.eth.getTransactionReceipt(transactionHash);
    return receipt;
  }

  async getBlockByNumber(blockNumber) {
    const block = await this.web3.eth.getBlock(blockNumber);
    return block;
  }

  async getSmartContract(contractAddress) {
    const contract = new this.web3.eth.Contract(ABI, contractAddress);
    return contract;
  }

  async callSmartContractMethod(contract, method, params) {
    const result = await contract.methods[method](...params).call();
    return result;
  }

  async sendSmartContractMethod(contract, method, params, fromAddress, value) {
    const gasPrice = await this.web3.eth.getGasPrice();
    const gas = await contract.methods[method](...params).estimateGas({ from: fromAddress, value });
    const transaction = {
      from: fromAddress,
      to: contract.options.address,
      data: contract.methods[method](...params).encodeABI(),
      value,
      gasPrice,
      gas,
    };
    const signedTransaction = await this.web3.eth.accounts.signTransaction(transaction, process.env.ETHEREUM_PRIVATE_KEY);
    const receipt = await this.web3.eth.sendSignedTransaction(signedTransaction.rawTransaction);
    return receipt;
  }
}

module.exports = EthereumAdapter;
