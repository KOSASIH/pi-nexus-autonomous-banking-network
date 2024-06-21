// piNetworkAnalytics.js
const express = require('express');
const app = express();
const Web3 = require('web3');
const piCoinStakingContract = require('../contracts/staking/PiCoinStaking.sol');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

app.use(express.json());

// API endpoint to retrieve staking analytics
app.get('/staking-analytics', async (req, res) => {
  const stakingData = await piCoinStakingContract.methods.getStakingData().call();
  res.json(stakingData);
});

// API endpoint to retrieve user staking history
app.get('/user-staking-history/:address', async (req, res) => {
  const address = req.params.address;
  const stakingHistory = await piCoinStakingContract.methods.getUserStakingHistory(address).call();
  res.json(stakingHistory);
});
