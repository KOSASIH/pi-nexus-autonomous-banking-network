const Web3 = require('web3');
const ethers = require('ethers');
const { Blockchain, Transaction } = require('blockchain-js');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const createWallet = () => {
  const wallet = ethers.Wallet.createRandom();
  return wallet;
};

const getBalance = (address) => {
  const balance = web3.eth.getBalance(address);
  return balance;
};

const sendTransaction = (from, to, amount) => {
  const tx = new Transaction(from, to, amount);
  const txHash = web3.eth.sendTransaction(tx);
  return txHash;
};

const getTransaction = (txHash) => {
  const tx = web3.eth.getTransaction(txHash);
  return tx;
};

const createSmartContract = (abi, bytecode) => {
  const contract = new web3.eth.Contract(abi);
  contract.deploy({
    data: bytecode,
  });
  return contract;
};

const callSmartContract = (contractAddress, functionName, args) => {
  const contract = new web3.eth.Contract(abi, contractAddress);
  const result = contract.methods[functionName](...args).call();
  return result;
};

module.exports = {
  createWallet,
  getBalance,
  sendTransaction,
  getTransaction,
  createSmartContract,
  callSmartContract,
};
