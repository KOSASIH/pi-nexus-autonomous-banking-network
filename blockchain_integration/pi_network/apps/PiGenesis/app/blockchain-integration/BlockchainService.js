import Blockchain from './Blockchain';

class BlockchainService {
  async getBalance(address) {
    return Blockchain.getBalance(address);
  }

  async getTransactionCount(address) {
    return Blockchain.getTransactionCount(address);
  }

  async sendTransaction(from, to, value) {
    return Blockchain.sendTransaction(from, to, value);
  }

  async deployContract() {
    return Blockchain.deployContract();
  }

  async getContractAddress() {
    return Blockchain.contractAddress;
  }

  async getContractABI() {
    return Blockchain.contractABI;
  }
}

export default BlockchainService;
