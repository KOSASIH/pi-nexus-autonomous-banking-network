import Web3 from 'web3';
import { Blockchain } from 'blockchain-js';

class Blockchain {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    this.contract = new this.web3.eth.Contract([
      {
        "constant": true,
        "inputs": [],
        "name": "getBalance",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
      }
    ], '0x...ContractAddress...');
  }

  getBalance(address) {
    return this.contract.methods.getBalance(address).call();
  }

  sendTransaction(from, to, value) {
    return this.web3.eth.sendTransaction({
      from,
      to,
      value,
      gas: '20000',
      gasPrice: this.web3.utils.toWei('20', 'gwei')
    });
  }
}

export default Blockchain;
