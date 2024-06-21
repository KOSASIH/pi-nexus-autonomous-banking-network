// piNetworkApi.js
const express = require('express');
const app = express();
const Web3 = require('web3');
const piCoinManagerContract = require('./PiCoinManager.sol');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

app.use(express.json());

// API endpoint to transfer Pi Coins
app.post('/transferPiCoins', async (req, res) => {
  const { from, to, amount } = req.body;
  try {
    const txCount = await web3.eth.getTransactionCount(from);
    const tx = {
      from,
      to,
      value: web3.utils.toWei(amount, 'ether'),
      gas: '20000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    res.json({ status: 'success', transactionHash: receipt.transactionHash });
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// API endpoint to mint new Pi Coins
app.post('/mintPiCoins', async (req, res) => {
  const { to, amount } = req.body;
  try {
    const tx = piCoinManagerContract.methods.mintPiCoins(to, amount).encodeABI();
    const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    res.json({ status: 'success', transactionHash: receipt.transactionHash });
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

app.listen(3000, () => {
  console.log('Pi Network API listening on port 3000');
});
