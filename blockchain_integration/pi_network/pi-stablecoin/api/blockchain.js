import { Blockchain } from 'bitcoinjs-lib';
import { ethers } from 'ethers.js';
import { KEVM } from 'kevm';
import { JAAK } from 'jaak';
import { MerkleTree } from 'merkletreejs';
import { elliptic } from 'elliptic';
import { scrypt } from 'scrypt-js';

class PiBlockchain extends Blockchain {
  constructor() {
    super();
    this.chain = [];
    this.pendingTransactions = [];
    this.merkleTree = new MerkleTree();
    this.ecdsa = new elliptic.ec('secp256k1');
    this.scryptParams = {
      N: 16384,
      r: 8,
      p: 1,
    };
  }

  createTransaction(from, to, amount) {
    const transaction = {
      from,
      to,
      amount,
      timestamp: Date.now(),
      nonce: this.getNonce(from),
    };
    this.pendingTransactions.push(transaction);
    return transaction;
  }

  getNonce(address) {
    const transactions = this.chain.filter((tx) => tx.from === address);
    return transactions.length;
  }

  mineBlock() {
    const block = {
      transactions: this.pendingTransactions,
      previousBlockHash: this.getLastBlockHash(),
      timestamp: Date.now(),
    };
    this.chain.push(block);
    this.pendingTransactions = [];
    return block;
  }

  getLastBlockHash() {
    return this.chain.length === 0 ? '0x0000000000000000000000000000000000000000000000000000000000000000' : this.chain[this.chain.length - 1].hash;
  }

  getBlockHash(block) {
    const blockString = JSON.stringify(block);
    const hash = this.ecdsa.hash(blockString);
    return hash;
  }

  verifyTransaction(transaction) {
    const signature = transaction.signature;
    const publicKey = this.ecdsa.recover(transaction.from, signature);
    return publicKey === transaction.from;
  }

  addTransactionToMerkleTree(transaction) {
    this.merkleTree.add(transaction.hash);
  }

  getMerkleRoot() {
    return this.merkleTree.getRoot();
  }

  scryptHash(data) {
    return scrypt.hash(data, this.scryptParams);
  }

  async deploySmartContract(contractCode) {
    const compiledContract = await KEVM.compile(contractCode);
    const deployedContract = await JAAK.deployContract(compiledContract);
    return deployedContract;
  }

  async executeSmartContract(contractAddress, functionName, functionArgs) {
    const contractInstance = await JAAK.getContractInstance(contractAddress);
    const functionResult = await contractInstance[functionName](...functionArgs);
    return functionResult;
  }
}

export { PiBlockchain as Blockchain };
