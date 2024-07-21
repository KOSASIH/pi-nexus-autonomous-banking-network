// identity-verification-service.js
const Web3 = require('web3');
const identityVerificationContract = require('./identity-verification-contract');

class IdentityVerificationService {
  constructor() {
    this.web3 = new Web3(
      new Web3.providers.HttpProvider(
        'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
      ),
    );
    this.contract = new this.web3.eth.Contract(
      identityVerificationContract.abi,
      identityVerificationContract.address,
    );
  }

  async verifyUserIdentity(userId, name, email, password) {
    const txCount = await this.web3.eth.getTransactionCount(userId);
    const tx = {
      from: userId,
      to: this.contract.address,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      data: this.contract.methods
        .verifyUserIdentity(name, email, password)
        .encodeABI(),
    };
    const signedTx = await this.web3.eth.accounts.signTransaction(
      tx,
      '0x' + userId,
    );
    await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  }

  async getUserIdentity(userId) {
    const result = await this.contract.methods.getUserIdentity(userId).call();
    return result;
  }
}

module.exports = IdentityVerificationService;
