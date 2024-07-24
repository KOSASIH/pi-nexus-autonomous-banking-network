import * as Web3 from 'web3';
import * as Ethereum from 'ethereumjs-tx';

class AstralPlaneBlockchain {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/454c372bb595486f90fc6295b128695c'));
    this.ethereum = Ethereum;
  }

  async createAsset(asset) {
    const txCount = await this.web3.eth.getTransactionCount('0xYourAddress');
    const tx = {
      from: '0xYourAddress',
      to: '0xContractAddress',
      value: Web3.utils.toWei('0.01', 'ether'),
      gas: '20000',
      gasPrice: Web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: this.ethereum.contract.methods.createAsset(asset.name, asset.description, asset.image, asset.price).encodeABI(),
    };
    const signedTx = await this.ethereum.signTransaction(tx, '0xYourPrivateKey');
    await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  }

  async getAsset(assetId) {
    const asset = await this.ethereum.contract.methods.getAsset(assetId).call();
    return asset;
  }

  async transferAsset(assetId, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount('0xYourAddress');
    const tx = {
      from: '0xYourAddress',
      to: '0xContractAddress',
      value: Web3.utils.toWei('0.01', 'ether'),
      gas: '20000',
      gasPrice: Web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: this.ethereum.contract.methods.transferAsset(assetId, toAddress).encodeABI(),
    };
    const signedTx = await this.ethereum.signTransaction(tx, '0xYourPrivateKey');
    await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  }
}

export default AstralPlaneBlockchain;
