import Web3 from 'web3';
import { curveABI } from './curveABI';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const curveContractAddress = '0x...';
const curveContract = new web3.eth.Contract(curveABI, curveContractAddress);

export async function getCurveAPY(tokenAddress) {
  const apy = await curveContract.methods.getAPY(tokenAddress).call();
  return apy;
}

export async function getCurveBalance(address) {
  const balance = await curveContract.methods.balanceOf(address).call();
  return balance;
}

export async function depositToCurve(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: curveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}

export async function withdrawFromCurve(address, amount) {
  const txCount = await web3.eth.getTransactionCount(address);
  const tx = {
    from: address,
    to: curveContractAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt;
}
