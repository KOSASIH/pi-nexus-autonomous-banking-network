import Web3 from 'web3';

class TokenContract {
  constructor(sidraChainSDK, address) {
    this.sidraChainSDK = sidraChainSDK;
    this.address = address;
    this.contract = new sidraChainSDK.web3.eth.Contract([
      {
        "constant": true,
        "inputs": [],
        "name": "balanceOf",
        "outputs": [
          {
            "name": "",
            "type": "uint256"
          }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
      },
      {
        "constant": false,
        "inputs": [
          {
            "name": "_to",
            "type": "address"
          },
          {
            "name": "_value",
            "type": "uint256"
          }
        ],
        "name": "transfer",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
      }
    ], address);
  }

  async deployed() {
    return this.contract.deployed();
  }

  async balanceOf(account) {
    return this.contract.methods.balanceOf(account).call();
  }

  async transfer(to, value) {
    return this.contract.methods.transfer(to, value).send({ from: this.sidraChainSDK.defaultAccount });
  }
}

export { TokenContract };
