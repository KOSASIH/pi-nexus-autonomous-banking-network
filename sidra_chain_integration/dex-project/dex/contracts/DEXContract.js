import Web3 from 'web3';

class DEXContract {
  constructor(sidraChainSDK, address) {
    this.sidraChainSDK = sidraChainSDK;
    this.address = address;
    this.contract = new sidraChainSDK.web3.eth.Contract([
      {
        "constant": true,
        "inputs": [],
        "name": "getOrders",
        "outputs": [
          {
            "name": "",
            "type": "uint256[]"
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
            "name": "_amount",
            "type": "uint256"
          },
          {
            "name": "_price",
            "type": "uint256"
          }
        ],
        "name": "placeOrder",
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

  async getOrders() {
    return this.contract.methods.getOrders().call();
  }

  async placeOrder(amount, price) {
    return this.contract.methods.placeOrder(amount, price).send({ from: this.sidraChainSDK.defaultAccount });
  }
}

export { DEXContract };
