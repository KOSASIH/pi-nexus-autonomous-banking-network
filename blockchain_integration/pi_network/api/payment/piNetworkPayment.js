// piNetworkPayment.js
const express = require('express');
const app = express();
const Web3 = require('web3');
const piCoinLendingContract = require('../contracts/lending/PiCoinLending.sol');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

app.use(express.json());

// API endpoint to process payments
app.post('/process-payment', async (req, res)=> {
  const { amount, recipient } = req.body;
  const paymentTx = await piCoinLendingContract.methods.processPayment(amount, recipient).call();
  res.json({ paymentTx });
});

// API endpoint to retrieve payment history
app.get('/payment-history/:address', async (req, res) => {
  const address = req.params.address;
  const paymentHistory = await piCoinLendingContract.methods.getPaymentHistory(address).call();
  res.json(paymentHistory);
});
