const Web3 = require('web3');
const Ethers = require('ethers');

class AtomicSwapContract {
  constructor(atomicSwapContractAddress, providerUrl) {
    this.atomicSwapContractAddress = atomicSwapContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async executeSwap(swapId, senderAddress, recipientAddress, amount) {
    const txCount = await this.web3.eth.getTransactionCount(this.atomicSwapContractAddress);
    const tx = {
      from: this.atomicSwapContractAddress,
      to: senderAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getBalance(address) {
    return this.web3.eth.getBalance(address);
  }
}

module.exports = AtomicSwapContract;
