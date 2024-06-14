// Import dependencies and utilities
import { createAction, createAsyncThunk } from '@reduxjs/toolkit';
import { api } from '../api';
import { constants } from '../utils/constants';
import { encrypt, decrypt } from '../utils/crypto';
import { getWeb3 } from '../utils/web3';

// Action types
const PORTFOLIO_LOAD = 'portfolio/LOAD';
const PORTFOLIO_UPDATE = 'portfolio/UPDATE';
const PORTFOLIO_TRANSFER = 'portfolio/TRANSFER';
const PORTFOLIO_STAKE = 'portfolio/STAKE';
const PORTFOLIO_UNSTAKE = 'portfolio/UNSTAKE';
const PORTFOLIO_CLAIM_REWARDS = 'portfolio/CLAIM_REWARDS';

// Action creators
export const loadPortfolio = createAsyncThunk(PORTFOLIO_LOAD, async (address) => {
  const web3 = getWeb3();
  const portfolioContract = new web3.eth.Contract(constants.PORTFOLIO_CONTRACT_ABI, constants.PORTFOLIO_CONTRACT_ADDRESS);
  const portfolioData = await portfolioContract.methods.getPortfolio(address).call();
  return encrypt(portfolioData);
});

export const updatePortfolio = createAction(PORTFOLIO_UPDATE, (updates) => {
  return { payload: encrypt(updates) };
});

export const transferAssets = createAsyncThunk(PORTFOLIO_TRANSFER, async (transferData) => {
  const web3 = getWeb3();
  const portfolioContract = new web3.eth.Contract(constants.PORTFOLIO_CONTRACT_ABI, constants.PORTFOLIO_CONTRACT_ADDRESS);
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: transferData.from,
    to: transferData.to,
    value: web3.utils.toWei(transferData.amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount,
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, constants.PRIVATE_KEY);
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt.transactionHash;
});

export const stakeAssets = createAsyncThunk(PORTFOLIO_STAKE, async (stakeData) => {
  const web3 = getWeb3();
  const stakingContract = new web3.eth.Contract(constants.STAKING_CONTRACT_ABI, constants.STAKING_CONTRACT_ADDRESS);
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: stakeData.from,
    value: web3.utils.toWei(stakeData.amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount,
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, constants.PRIVATE_KEY);
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt.transactionHash;
});

export const unstakeAssets = createAsyncThunk(PORTFOLIO_UNSTAKE, async (unstakeData) => {
  const web3 = getWeb3();
  const stakingContract = new web3.eth.Contract(constants.STAKING_CONTRACT_ABI, constants.STAKING_CONTRACT_ADDRESS);
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: unstakeData.from,
    value: web3.utils.toWei(unstakeData.amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount,
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, constants.PRIVATE_KEY);
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt.transactionHash;
});

export const claimRewards = createAsyncThunk(PORTFOLIO_CLAIM_REWARDS, async (claimData) => {
  const web3 = getWeb3();
  const rewardsContract = new web3.eth.Contract(constants.REWARDS_CONTRACT_ABI, constants.REWARDS_CONTRACT_ADDRESS);
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: claimData.from,
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount,
  };
  const signedTx = await web3.eth.accounts.signTransaction(tx, constants.PRIVATE_KEY);
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  return receipt.transactionHash;
});
