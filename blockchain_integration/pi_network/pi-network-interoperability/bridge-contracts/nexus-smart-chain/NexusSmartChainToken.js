const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class NexusSmartChainToken {
  constructor(tokenContractAddress, providerUrl) {
    this.tokenContractAddress = tokenContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async transferTokens(amount, fromAddress, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.tokenContractAddress);
    const tx = {
      from: this.tokenContractAddress,
      to: this.tokenContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${this.encodeTransferTokens(amount, fromAddress, toAddress)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getBalance(address) {
    return this.api.query.nexusSmartChainToken.balanceOf(address);
  }

  encodeTransferTokens(amount, fromAddress, toAddress) {
    return this.web3.eth.abi.encodeFunctionCall({
      name: 'transfer',
      type: 'function',
      inputs: [{
        type: 'uint256',
        name: 'amount'
      }, {
        type: 'address',
        name: 'from'
      }, {
        type: 'address',
        name: 'to'
      }]
    }, [amount, fromAddress, toAddress]);
  }
}

module.exports = NexusSmartChainToken;
