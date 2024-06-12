import { Block } from './block';

class Blockchain {
  constructor() {
    this.chain = [this.createGenesisBlock()];
  }

  createGenesisBlock() {
    return new Block([], '0');
  }

  addBlock(block) {
    // implement block validation and addition logic
  }

  getLatestBlock() {
    return this.chain[this.chain.length - 1];
  }
}

export default Blockchain;
