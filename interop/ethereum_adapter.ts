import Web3 from 'web3';

class EthereumAdapter {
  private web3: Web3;

  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
  }

  async sendTransaction(transaction: any) {
    return this.web3.eth.sendTransaction(transaction);
  }

  async receiveTransaction(transaction: any) {
    return this.web3.eth.getTransactionReceipt(transaction.hash);
  }
}

export default EthereumAdapter;
