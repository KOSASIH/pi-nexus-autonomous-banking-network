const Web3 = require('web3');
const { ChainId, TokenAmount, Pair } = require('@uniswap/sdk');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const piNetworkSmartContractAddress = '0x...'; // Replace with your smart contract address
const piNetworkSmartContractABI = [...]; // Replace with your smart contract ABI

const piNetworkSmartContract = new web3.eth.Contract(piNetworkSmartContractABI, piNetworkSmartContractAddress);

async function createAccount(user, initialBalance) {
  const txCount = await web3.eth.getTransactionCount(user);
  const txData = piNetworkSmartContract.methods.createAccount(user, initialBalance).encodeABI();
  const tx = {
    from: user,
    to: piNetworkSmartContractAddress,
    data: txData,
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with your private key
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

async function transferFunds(from, to, amount) {
  const txCount = await web3.eth.getTransactionCount(from);
  const txData = piNetworkSmartContract.methods.transferFunds(from, to, amount).encodeABI();
  const tx = {
    from: from,
    to: piNetworkSmartContractAddress,
    data: txData,
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with your private key
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

async function getAccountBalance(user) {
  const balance = await piNetworkSmartContract.methods.getAccountBalance(user).call();
  return balance;
}

async function getTransactionHistory(user) {
  const transactionHistory = await piNetworkSmartContract.methods.getTransactionHistory(user).call();
  return transactionHistory;
}

module.exports = {
  createAccount,
  transferFunds,
  getAccountBalance,
  getTransactionHistory,
};
