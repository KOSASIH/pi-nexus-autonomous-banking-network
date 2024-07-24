import * as Web3 from 'web3';
import * as EthereumTx from 'ethereumjs-tx';

class AstralPlaneBlockchain {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    this.contractAddress = '0x...';
    this.contractABI = [...];
  }

  async getBalance(address) {
    const balance = await this.web3.eth.getBalance(address);
    return balance;
  }

  async transferETH(from, to, amount) {
    const txCount = await this.web3.eth.getTransactionCount(from);
    const tx = new EthereumTx({
      from,
      to,
      value: this.web3.utils.toWei(amount, 'ether'),
      gas: '20000',
      gasPrice: this.web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    });
    const signedTx = await this.web3.eth.accounts.signTransaction(tx, from);
    await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  }

  async createAsset(name, description, image, price) {
    const contract = new this.web3.eth.Contract(this.contractABI, this.contractAddress);
    const tx = contract.methods.createAsset(name, description, image, price).encodeABI();
    const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...');
    await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  }
}

export default AstralPlaneBlockchain;
