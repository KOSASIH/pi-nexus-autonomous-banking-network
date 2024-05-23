const Web3 = require('web3');
const piTokenAbi = require('./pi-token.json');

class PITokenService {
  constructor(web3, contractAddress) {
    this.web3 = web3;
    this.contract = new this.web3.eth.Contract(piTokenAbi, contractAddress);
  }

  async issue(amount) {
    const accounts = await this.web3.eth.getAccounts();
    await this.contract.methods.issue(amount).send({ from: accounts[0] });
  }

  async transfer(to, amount) {
    const accounts = await this.web3.eth.getAccounts();
    await this.contract.methods.transfer(to, amount).send({ from: accounts[0] });
  }

  async balanceOf(address) {
    return await this.contract.methods.balanceOf(address).call();
  }
}

module.exports = PITokenService;
