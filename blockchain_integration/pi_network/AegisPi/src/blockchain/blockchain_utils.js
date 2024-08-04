import { Crypto } from './crypto';

class BlockchainUtils {
  static calculateHash(index, previousHash, timestamp, data) {
    const dataString = JSON.stringify(data);
    const hash = Crypto.sha256(`${index}${previousHash}${timestamp}${dataString}`);
    return hash;
  }

  static validateProofOfWork(hash, difficulty) {
    const hashBuffer = Buffer.from(hash, 'hex');
    const requiredPrefix = Buffer.alloc(difficulty, 0);
    if (hashBuffer.slice(0, difficulty).equals(requiredPrefix)) {
      return true;
    }
    return false;
  }

  static async mineBlock(block, difficulty) {
    let nonce = 0;
    while (true) {
      const hash = BlockchainUtils.calculateHash(block.index, block.previousHash, block.timestamp, block.data);
      if (BlockchainUtils.validateProofOfWork(hash, difficulty)) {
        block.hash = hash;
        return block;
      }
      nonce++;
      block.timestamp = Date.now();
    }
  }
}

export { BlockchainUtils };
