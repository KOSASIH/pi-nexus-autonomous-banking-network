import { elliptic } from 'elliptic';
import { scrypt } from 'scrypt-js';
import { MerkleTree } from 'merkletreejs';
import { Transaction } from './Transaction';

class Block {
  constructor(transactions, previousBlockHash, nonce = 0) {
    this.transactions = transactions;
    this.previousBlockHash = previousBlockHash;
    this.nonce = nonce;
    this.timestamp = Date.now();
    this.hash = this.calculateHash();
    this.merkleTree = this.createMerkleTree();
  }

  calculateHash() {
    const blockString = JSON.stringify(this);
    const hash = elliptic.hash(blockString);
    return hash;
  }

  createMerkleTree() {
    const merkleTree = new MerkleTree();
    this.transactions.forEach((transaction) => {
      merkleTree.add(transaction.hash);
    });
    return merkleTree;
  }

  getMerkleRoot() {
    return this.merkleTree.getRoot();
  }

  addTransaction(transaction) {
    this.transactions.push(transaction);
    this.merkleTree.add(transaction.hash);
    this.hash = this.calculateHash();
  }

  mineBlock(difficulty) {
    while (this.hash.substring(0, difficulty) !== Array(difficulty + 1).join('0')) {
      this.nonce++;
      this.hash = this.calculateHash();
    }
  }

  scryptHash(data) {
    return scrypt.hash(data, {
      N: 16384,
      r: 8,
      p: 1,
    });
  }

  verifyBlock() {
    if (!this.previousBlockHash) return true;
    const previousBlock = Block.getBlockByHash(this.previousBlockHash);
    if (!previousBlock) return false;
    if (previousBlock.hash !== this.previousBlockHash) return false;
    return true;
  }

  static getBlockByHash(hash) {
    // Retrieve the block from the blockchain or a local storage
    // For demonstration purposes, we'll just return a dummy block
    return new Block([], hash);
  }
}

export default Block;
