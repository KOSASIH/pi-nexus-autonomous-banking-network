const Web3 = require('web3');
const Ethers = require('ethers');

class PolkadotERC20Contract {
  constructor(erc20ContractAddress, providerUrl) {
    this.erc20ContractAddress = erc20ContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async transfer(amount, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.erc20ContractAddress);
    const tx = {
      from: this.erc20ContractAddress,
      to: this.erc20ContractAddress,
      value: amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0xa9059cbb${toAddress.toString(16)}${amount.toString(16)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async balanceOf(address) {
    return this.web3.eth.call({
      to: this.erc20ContractAddress,
      data: `0x70a08231000000000000000000000000${address.toString(16)}`
    });
  }
}

module.exports = PolkadotERC20Contract;
