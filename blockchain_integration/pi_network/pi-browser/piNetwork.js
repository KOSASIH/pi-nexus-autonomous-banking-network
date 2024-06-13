import { Blockchain } from './blockchain';
import { Miner } from './miner';
import { Wallet } from './wallet';

class PINetwork {
  constructor() {
    this.blockchain = new Blockchain();
    this.miner = new Miner(this.blockchain);
    this.wallet = new Wallet('privateKey');
  }

  connectToNetwork() {
    // implement network connection logic
  }

  getBlockchain() {
    return this.blockchain;
  }

  getWalletBalance() {
    return this.wallet.getBalance();
  }

  sendTransaction(recipient, amount) {
    this.wallet.sendTransaction(recipient, amount);
  }

  mineBlock() {
    this.miner.mineBlock([]);
  }
}

export default PINetwork;
