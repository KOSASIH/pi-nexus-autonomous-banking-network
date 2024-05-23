const Web3 = require('web3');
const piTokenContractArtifact = require('./pi-token.json');

class PiTokenContractWrapper {
  constructor(web3, contractAddress) {
    this.web3 = web3;
    this.contract = new web3.eth.Contract(
      piTokenContractArtifact.abi,
      contractAddress
    );
  }

  async getName() {
    return await this.contract.methods.name().call();
  }

  async getSymbol() {
    return await this.contract.methods.symbol().call();
  }

  async getDecimals() {
    return await this.contract.methods.decimals().call();
  }

  async getTotalSupply() {
    return await this.contract.methods.totalSupply().call();
  }

  async getBalance(address) {
    return await this.contract.methods.balanceOf(address).call();
  }

  async transfer(to, value) {
    const accounts = await this.web3.eth.getAccounts();
    return await this.contract.methods.transfer(to, value).send({ from: accounts[0] });
  }

  async approve(spender, value) {
    const accounts = await this.web3.eth.getAccounts();
    return await this.contract.methods.approve(spender, value).send({ from: accounts[0] });
  }

  async transferFrom(from, to, value) {
    const accounts = await this.web3.eth.getAccounts();
    return await this.contract.methods.transferFrom(from, to, value).send({ from: accounts[0] });
  }
}

module.exports = PiTokenContractWrapper;
