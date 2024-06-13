import { Blockchain } from './blockchain';
import { Block } from './block';

class Miner {
  constructor(blockchain) {
    this.blockchain = blockchain;
  }

  mineBlock(transactions) {
    const block = new Block(transactions, this.blockchain.getLatestBlock().hash);
    block.mine(3); // adjust difficulty as needed
    this.blockchain.addBlock(block);
  }
}

export default Miner;
