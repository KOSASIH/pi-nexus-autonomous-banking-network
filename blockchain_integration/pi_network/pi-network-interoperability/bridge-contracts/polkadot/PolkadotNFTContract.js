const Web3 = require('web3');
const Ethers = require('ethers');
const { ApiPromise, WsProvider } = require('@polkadot/api');

class PolkadotNFTContract {
  constructor(polkadotNFTContractAddress, providerUrl) {
    this.polkadotNFTContractAddress = polkadotNFTContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
    this.api = new ApiPromise({
      provider: new WsProvider(providerUrl)
    });
  }

  async mintNFT(tokenId, ownerAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotNFTContractAddress);
    const tx = {
      from: this.polkadotNFTContractAddress,
      to: this.polkadotNFTContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${tokenId.toString(16)}${ownerAddress.slice(2)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async transferNFT(tokenId, fromAddress, toAddress) {
    const txCount = await this.web3.eth.getTransactionCount(this.polkadotNFTContractAddress);
    const tx = {
      from: fromAddress,
      to: this.polkadotNFTContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${tokenId.toString(16)}${toAddress.slice(2)}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getNFTOwner(tokenId) {
    return this.api.query.nft.ownerOf(tokenId);
  }
}

module.exports = PolkadotNFTContract;
