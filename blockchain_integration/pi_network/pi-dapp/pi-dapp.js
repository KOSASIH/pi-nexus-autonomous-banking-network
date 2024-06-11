import Web3 from 'web3';
import { abi, bytecode } from './PiNetworkContract';

const rpcEndpoint = 'https://go.getblock.io/307538472a884879b4cfd275a0d12b1c';
const web3 = new Web3(new Web3.providers.HttpProvider(rpcEndpoint));

const contractAddress = '0x...'; // Replace with the deployed contract address
const contract = new web3.eth.Contract(abi, contractAddress);

// Function to send Pi coins
async function sendPiCoins(toAddress, amount) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: '0x...', // Replace with the sender's address
    to: toAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's private key
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

// Function to get Pi coin balance
async function getPiCoinBalance(address) {
  const balance = await contract.methods.balanceOf(address).call();
  return balance;
}

// Function to transfer Pi coins
async function transferPiCoins(fromAddress, toAddress, amount) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: fromAddress,
    to: toAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's private key
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

export { sendPiCoins, getPiCoinBalance, transferPiCoins };
