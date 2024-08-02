import express from 'express';
import { json } from 'body-parser';
import { Blockchain } from './blockchain';
import { Wallet } from './wallet';
import { ethers } from 'ethers.js';
import { KEVM } from 'kevm';
import { JAAK } from 'jaak';

const app = express();
app.use(json());

const blockchain = new Blockchain();
const wallet = new Wallet();
const ethersProvider = new ethers.providers.JsonRpcProvider();
const kevm = new KEVM();
const jaak = new JAAK();

app.post('/transactions', async (req, res) => {
  const { from, to, amount } = req.body;
  const transaction = await blockchain.createTransaction(from, to, amount);
  const signedTransaction = await wallet.signTransaction(transaction);
  const gasEstimate = await ethersProvider.estimateGas(signedTransaction);
  const transactionReceipt = await ethersProvider.sendTransaction(signedTransaction);
  res.json(transactionReceipt);
});

app.get('/balance/:address', async (req, res) => {
  const address = req.params.address;
  const balance = await wallet.getBalance(address);
  res.json(balance);
});

app.post('/deployContract', async (req, res) => {
  const { contractCode } = req.body;
  const compiledContract = await kevm.compile(contractCode);
  const deployedContract = await jaak.deployContract(compiledContract);
  res.json(deployedContract);
});

app.post('/executeContract', async (req, res) => {
  const { contractAddress, functionName, functionArgs } = req.body;
  const contractInstance = await jaak.getContractInstance(contractAddress);
  const functionResult = await contractInstance[functionName](...functionArgs);
  res.json(functionResult);
});

app.listen(3000, () => {
  console.log('Pi Stablecoin API listening on port 3000');
});
