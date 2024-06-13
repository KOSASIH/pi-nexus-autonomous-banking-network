import { Ethers } from 'ethers';

class BlockchainExplorer {
  constructor() {
    this.ethers = new Ethers();
  }

  async getBlockByNumber(blockNumber) {
    const block = await this.ethers.getBlockByNumber(blockNumber);
    return block;
  }

  async getTransactionByHash(transactionHash) {
    const transaction = await this.ethers.getTransactionByHash(transactionHash);
    return transaction;
  }
}

export default BlockchainExplorer;
