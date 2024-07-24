import express from 'express';
import Web3 from 'web3';
import { AstralPlaneToken } from '../contracts/AstralPlaneToken';

const app = express();
const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

app.get('/api/astralplane/balance', async (req, res) => {
  const account = req.query.account;
  const balance = await web3.eth.getBalance(account);
  res.json({ balance });
});

app.get('/api/astralplane/tokenbalance', async (req, res) => {
  const account = req.query.account;
  const tokenBalance = await AstralPlaneToken.balanceOf(account);
  res.json({ tokenBalance });
});

app.listen(3000, () => {
  console.log('AstralPlane API listening on port 3000');
});
