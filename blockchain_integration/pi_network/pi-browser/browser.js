import { PINetwork } from './piNetwork';

class Browser {
  constructor(piNetwork) {
    this.piNetwork = piNetwork;
  }

  connectToNetwork() {
    this.piNetwork.connectToNetwork();
  }

  getBlockchain() {
    return this.piNetwork.getBlockchain();
  }

  getWalletBalance() {
    return this.piNetwork.getWalletBalance();
  }

  sendTransaction(recipient, amount) {
    this.piNetwork.sendTransaction(recipient, amount);
  }

  mineBlock() {
    this.piNetwork.mineBlock();
  }
}

export default Browser;
