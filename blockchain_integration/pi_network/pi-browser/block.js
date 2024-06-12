import { hash } from 'hash-sum';
import { Transaction } from './transaction';

class Block {
  constructor(transactions, previousBlockHash) {
    this.transactions = transactions;
    this.previousBlockHash = previousBlockHash;
    this.timestamp = Date.now();
    this.hash = hash(this.toString());
  }

  toString() {
    return `${this.transactions.map(tx => tx.toString()).join(',')}:${this.previousBlockHash}:${this.timestamp}`;
  }

  mine(difficulty) {
    // implement block mining logic
  }
}

export default Block;
