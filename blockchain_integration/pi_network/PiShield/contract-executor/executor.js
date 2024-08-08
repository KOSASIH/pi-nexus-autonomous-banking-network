// executor.js

const Web3 = require('web3');
const { Contract } = require('web3-eth-contract');

class Executor {
  constructor(config) {
    this.config = config;
    this.web3 = new Web3(new Web3.providers.HttpProvider(config.providerUrl));
    this.contract = new Contract(config.contractAddress, config.abi);
  }

  async executeFunction(functionName, params) {
    const functionSignature = this.contract.methods[functionName].encodeABI();
    const txCount = await this.web3.eth.getTransactionCount(this.config.fromAddress);
    const tx = {
      from: this.config.fromAddress,
      to: this.config.contractAddress,
      data: functionSignature + this.encodeParams(params),
      gas: this.config.gas,
      gasPrice: this.config.gasPrice
    };
    const signedTx = await this.web3.eth.accounts.signTransaction(tx, this.config.privateKey);
    const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt;
  }

  encodeParams(params) {
    const encodedParams = [];
    for (const param of params) {
      if (typeof param === 'string') {
        encodedParams.push(this.web3.utils.stringToHex(param));
      } else if (typeof param === 'number') {
        encodedParams.push(this.web3.utils.numberToHex(param));
      } else {
        throw new Error(`Unsupported parameter type: ${typeof param}`);
      }
    }
    return encodedParams.join('');
  }
}

module.exports = Executor;
