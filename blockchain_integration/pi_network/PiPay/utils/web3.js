// utils/web3.js
import Web3 from 'web3';
import { ethers } from 'ethers';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractABI = [
  {
    "constant": true,
    "inputs": [],
    "name": "getMerchantAddress",
    "outputs": [
      {
        "name": "",
        "type": "address"
      }
    ],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  },
  {
    "constant": true,
    "inputs": [],
    "name": "getBalance",
    "outputs": [
      {
        "name": "",
        "type": "uint256"
      }
    ],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  },
  {
    "constant": false,
    "inputs": [
      {
        "name": "_payer",
        "type": "address"
      },
      {
        "name": "_payee",
        "type": "address"
      },
      {
        "name": "_amount",
        "type": "uint256"
      }
    ],
    "name": "processPayment",
    "outputs": [],
    "payable": false,
    "stateMutability": "nonpayable",
    "type": "function"
  }
];

const contractAddress = '0x...YOUR_CONTRACT_ADDRESS...';

const paymentGatewayContract = new web3.eth.Contract(contractABI, contractAddress);

const getMerchantAddress = async (publicKey) => {
  return paymentGatewayContract.methods.getMerchantAddress(publicKey).call();
};

const getBalance = async (publicKey) => {
  return paymentGatewayContract.methods.getBalance(publicKey).call();
};

const processPayment = async (payer, payee, amount) => {
  return paymentGatewayContract.methods.processPayment(payer, payee, amount).send({ from: payer });
};

const getTransactionCount = async (address) => {
  return web3.eth.getTransactionCount(address);
};

const sendTransaction = async (tx) => {
  return web3.eth.sendTransaction(tx);
};

const getTransactionReceipt = async (txHash) => {
  return web3.eth.getTransactionReceipt(txHash);
};

const ethersProvider = new ethers.providers.JsonRpcProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID');

const getBlockNumber = async () => {
  return ethersProvider.getBlockNumber();
};

const getBlock = async (blockNumber) => {
  return ethersProvider.getBlock(blockNumber);
};

export {
  web3,
  paymentGatewayContract,
  getMerchantAddress,
  getBalance,
  processPayment,
  getTransactionCount,
  sendTransaction,
  getTransactionReceipt,
  ethersProvider,
  getBlockNumber,
  getBlock,
};
