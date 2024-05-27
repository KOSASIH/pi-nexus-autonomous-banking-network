const Web3 = require("web3");
const piBankContractArtifact = require("./pi-bank.json");

class PiBankContractWrapper {
  constructor(web3, contractAddress) {
    this.web3 = web3;
    this.contract = new web3.eth.Contract(
      piBankContractArtifact.abi,
      contractAddress,
    );
  }

  async deposit(amount) {
    const accounts = await this.web3.eth.getAccounts();
    return await this.contract.methods
      .deposit()
      .send({ from: accounts[0], value: amount });
  }

  async withdraw(amount) {
    const accounts = await this.web3.eth.getAccounts();
    return await this.contract.methods
      .withdraw(amount)
      .send({ from: accounts[0] });
  }

  async canWithdraw() {
    return await this.contract.methods.canWithdraw().call();
  }

  async getBalance() {
    return await this.contract.methods.getBalance().call();
  }
}

module.exports = PiBankContractWrapper;
