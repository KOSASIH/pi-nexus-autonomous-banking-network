// app.js
const express = require('express');
const app = express();
const BankingPlatform = require('./blockchain_integration/banking_platform');

const bankingPlatform = new BankingPlatform();

app.post('/create-account', async (req, res) => {
  const userIdentity = req.body.user_identity;
  const account = await bankingPlatform.createAccount(userIdentity);
  res.json(account);
});

app.post('/deposit-funds', async (req, res) => {
  const accountAddress = req.body.account_address;
  const amount = req.body.amount;
  await bankingPlatform.depositFunds(accountAddress, amount);
  res.json({ message: 'Funds deposited successfully' });
});

app.post('/withdraw-funds', async (req, res) => {
  const accountAddress = req.body.account_address;
  const amount = req.body.amount;
  await bankingPlatform.withdrawFunds(accountAddress, amount);
  res.json({ message: 'Funds withdrawn successfully' });
});

app.get('/get-balance', async (req, res) => {
  const accountAddress = req.query.account_address;
  const balance = await bankingPlatform.getBalance(accountAddress);
  res.json({ balance });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
