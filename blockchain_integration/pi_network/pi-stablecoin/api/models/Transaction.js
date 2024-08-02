import { elliptic } from 'elliptic';
import { scrypt } from 'scrypt-js';
import { MerkleTree } from 'merkletreejs';

class Transaction {
  constructor(from, to, amount, data = '') {
    this.from = from;
    this.to = to;
    this.amount = amount;
    this.data = data;
    this.timestamp = Date.now();
    this.nonce = this.getNonce(from);
    this.hash = this.calculateHash();
    this.signature = null;
  }

  getNonce(address) {
    // Get the nonce from the blockchain or a local storage
    // For demonstration purposes, we'll just increment a counter
    let nonce = 0;
    // ...
    return nonce;
  }

  calculateHash() {
    const txString = JSON.stringify(this);
    const hash = elliptic.hash(txString);
    return hash;
  }

  signTransaction(privateKey) {
    const signature = elliptic.sign(this.hash, privateKey);
    this.signature = signature;
    return signature;
  }

  verifyTransaction() {
    const publicKey = elliptic.recover(this.from, this.signature);
    return publicKey === this.from;
  }

  addTransactionToMerkleTree(merkleTree) {
    merkleTree.add(this.hash);
  }

  getMerkleProof(merkleTree) {
    return merkleTree.getProof(this.hash);
  }

  scryptHash(data) {
    return scrypt.hash(data, {
      N: 16384,
      r: 8,
      p: 1,
    });
  }
}

export default Transaction;
