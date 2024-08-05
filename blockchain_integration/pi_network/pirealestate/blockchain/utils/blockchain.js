const { ethers } = require("hardhat");
const { BigNumber } = require("ethers");
const { Block } = require("@ethereumjs/block");
const { Blockchain } = require("@ethereumjs/blockchain");
const { Transaction } = require("@ethereumjs/tx");
const { Wallet } = require("@ethereumjs/wallet");
const { Common } = require("@ethereumjs/common");
const { RLP } = require("@ethereumjs/rlp");
const { keccak256 } = require("keccak256");

// Set up the blockchain configuration
const blockchainConfig = {
  chainId: 1,
  networkId: 1,
  gasPrice: BigNumber.from("20.0"),
  gasLimit: 8000000,
};

// Create a new blockchain instance
const blockchain = new Blockchain({
  chainId: blockchainConfig.chainId,
  networkId: blockchainConfig.networkId,
  gasPrice: blockchainConfig.gasPrice,
  gasLimit: blockchainConfig.gasLimit,
});

// Create a new wallet instance
const wallet = new Wallet({
  privateKey: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
});

// Create a new block instance
const block = new Block({
  number: 1,
  hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  parentHash: "0x0000000000000000000000000000000000000000000000000000000000000000",
  timestamp: Date.now(),
  gasLimit: blockchainConfig.gasLimit,
  gasUsed: 0,
  transactions: [],
});

// Create a new transaction instance
const transaction = new Transaction({
  from: wallet.address,
  to: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  value: BigNumber.from("1.0"),
  gasPrice: blockchainConfig.gasPrice,
  gasLimit: blockchainConfig.gasLimit,
});

// Sign the transaction with the wallet's private key
const signedTransaction = wallet.signTransaction(transaction);

// Add the signed transaction to the block
block.transactions.push(signedTransaction);

// Calculate the block's hash
const blockHash = keccak256(RLP.encode(block.header));

// Add the block to the blockchain
blockchain.addBlock(block);

// Get the blockchain's current state
const state = blockchain.getState();

// Get the blockchain's current block number
const blockNumber = blockchain.getBlockNumber();

// Get the blockchain's current block hash
const currentBlockHash = blockchain.getBlockHash();

// Get the blockchain's current transaction count
const transactionCount = blockchain.getTransactionCount();

// Get the blockchain's current gas price
const gasPrice = blockchain.getGasPrice();

// Get the blockchain's current gas limit
const gasLimit = blockchain.getGasLimit();

// Export the blockchain functions
module.exports = {
  getBlockchainState: () => state,
  getBlockNumber: () => blockNumber,
  getBlockHash: () => currentBlockHash,
  getTransactionCount: () => transactionCount,
  getGasPrice: () => gasPrice,
  getGasLimit: () => gasLimit,
  addBlock: (block) => blockchain.addBlock(block),
  getBlock: (blockNumber) => blockchain.getBlock(blockNumber),
  getTransaction: (transactionHash) => blockchain.getTransaction(transactionHash),
  signTransaction: (transaction) => wallet.signTransaction(transaction),
};
