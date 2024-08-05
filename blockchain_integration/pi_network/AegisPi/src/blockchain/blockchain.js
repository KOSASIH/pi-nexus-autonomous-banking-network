import { BlockchainUtils } from './blockchain_utils';
import { Crypto } from './crypto';

class Block {
  constructor(index, previousHash, timestamp, data, hash) {
    this.index = index;
    this.previousHash = previousHash;
    this.timestamp = timestamp;
    this.data = data;
    this.hash = hash;
  }
}

class Blockchain {
  constructor() {
    this.chain = [this.createGenesisBlock()];
    this.pendingTransactions = [];
    this.miningDifficulty = 3;
    this.blockSizeLimit = 10;
  }

  createGenesisBlock() {
    return new Block(0, '0', 0, 'Genesis Block', '0');
  }

  getLatestBlock() {
    return this.chain[this.chain.length - 1];
  }

  addTransaction(transaction) {
    this.pendingTransactions.push(transaction);
  }

  minePendingTransactions(minerAddress) {
    const block = this.createBlock(this.pendingTransactions, minerAddress);
    this.chain.push(block);
    this.pendingTransactions = [];
    return block;
  }

  createBlock(transactions, minerAddress) {
    const previousBlock = this.getLatestBlock();
    const index = previousBlock.index + 1;
    const timestamp = Date.now();
    const data = transactions;
    const hash = BlockchainUtils.calculateHash(index, previousBlock.hash, timestamp, data);
    return new Block(index, previousBlock.hash, timestamp, data, hash);
  }

  validateChain() {
    for (let i = 1; i < this.chain.length; i++) {
      const currentBlock = this.chain[i];
      const previousBlock = this.chain[i - 1];
      if (currentBlock.hash !== BlockchainUtils.calculateHash(currentBlock.index, previousBlock.hash, currentBlock.timestamp, currentBlock.data)) {
        return false;
      }
    }
    return true;
  }

  static async getBlockchainInstance() {
    const blockchain = new Blockchain();
    await blockchain.initialize();
    return blockchain;
  }

  async initialize() {
    // Initialize the blockchain by loading the chain from storage or creating a new one
  }
}

export { Blockchain, Block };
