import Web3 from 'web3';
import { aaveABI } from './aaveABI';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const aaveContractAddress = '0x...';
const aaveContract = new web3.eth.Contract(aaveABI, aaveContractAddress);

export async function getAaveLendingRate(tokenAddress) {
  const lendingRate = await aaveContract.methods.getLendingRate(tokenAddress).call();
  return lendingRate;
}

export async function getAaveBorrowRate(tokenAddress) {
  const borrowRate = await aaveContract.methods.getBorrowRate(tokenAddress).call();
  return borrowRate;
}

export async function depositToAave(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: aaveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

export async function withdrawFromAave(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: aaveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

export async function borrowFromAave(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: aaveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

export async function repayAaveLoan(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: aaveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}
