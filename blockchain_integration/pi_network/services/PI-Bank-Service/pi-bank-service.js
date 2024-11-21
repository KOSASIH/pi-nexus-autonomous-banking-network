const Web3 = require('web3')
const piBankAbi = require('./pi-bank.json')

class PIBankService {
  constructor (web3, contractAddress) {
    this.web3 = web3
    this.contract = new this.web3.eth.Contract(piBankAbi, contractAddress)
  }

  async deposit (amount) {
    const accounts = await this.web3.eth.getAccounts()
    await this.contract.methods
      .deposit()
      .send({ from: accounts[0], value: amount })
  }

  async withdraw (amount) {
    const accounts = await this.web3.eth.getAccounts()
    await this.contract.methods.withdraw(amount).send({ from: accounts[0] })
  }

  async transfer (to, amount) {
    const accounts = await this.web3.eth.getAccounts()
    await this.contract.methods
      .transfer(to, amount)
      .send({ from: accounts[0] })
  }

  async getBalance () {
    const accounts = await this.web3.eth.getAccounts()
    return await this.contract.methods.getBalance().call({ from: accounts[0] })
  }
}

module.exports = PIBankService
