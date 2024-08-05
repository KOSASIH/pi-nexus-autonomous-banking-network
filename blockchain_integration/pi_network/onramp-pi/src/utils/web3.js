// onramp-pi/src/utils/web3.js

import Web3 from 'web3';
import { ethers } from 'ethers';
import { abi as ERC20_ABI } from '@openzeppelin/contracts/build/contracts/ERC20.json';
import { abi as UNISWAP_V2_ROUTER_ABI } from '@uniswap/v2-periphery/build/UniswapV2Router02.json';

const WEB3_PROVIDER_URL = 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID';
const WEB3_PROVIDER_NETWORK_ID = 1; // Mainnet
const UNISWAP_V2_ROUTER_ADDRESS = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D';
const WETH_ADDRESS = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2';

const web3 = new Web3(new Web3.providers.HttpProvider(WEB3_PROVIDER_URL));
const ethersProvider = new ethers.providers.JsonRpcProvider(WEB3_PROVIDER_URL);

const erc20Contract = new web3.eth.Contract(ERC20_ABI, WETH_ADDRESS);
const uniswapV2RouterContract = new web3.eth.Contract(UNISWAP_V2_ROUTER_ABI, UNISWAP_V2_ROUTER_ADDRESS);

async function getWalletBalance(address) {
  try {
    const balance = await web3.eth.getBalance(address);
    return web3.utils.fromWei(balance, 'ether');
  } catch (error) {
    console.error(error);
    throw new Error('Failed to get wallet balance');
  }
}

async function getERC20Balance(address, tokenAddress) {
  try {
    const balance = await erc20Contract.methods.balanceOf(address).call();
    return web3.utils.fromWei(balance, 'ether');
  } catch (error) {
    console.error(error);
    throw new Error('Failed to get ERC20 balance');
  }
}

async function approveERC20(tokenAddress, spenderAddress, amount) {
  try {
    const txCount = await web3.eth.getTransactionCount(tokenAddress);
    const tx = {
      from: tokenAddress,
      to: spenderAddress,
      value: web3.utils.toWei(amount, 'ether'),
      gas: '20000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt.transactionHash;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to approve ERC20');
  }
}

async function swapETHForERC20(amount, tokenAddress) {
  try {
    const path = [WETH_ADDRESS, tokenAddress];
    const amounts = [web3.utils.toWei(amount, 'ether'), 0];
    const deadline = Math.floor(Date.now() / 1000) + 60 * 20; // 20 minutes
    const tx = uniswapV2RouterContract.methods.swapExactETHForTokens(
      amounts,
      path,
      deadline
    );
    const txCount = await web3.eth.getTransactionCount(tokenAddress);
    const signedTx = await web3.eth.accounts.signTransaction(tx, 'YOUR_PRIVATE_KEY');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt.transactionHash;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to swap ETH for ERC20');
  }
}

async function getUniswapV2RouterQuote(amount, tokenAddress) {
  try {
    const path = [WETH_ADDRESS, tokenAddress];
    const amounts = [web3.utils.toWei(amount, 'ether'), 0];
    const quote = await uniswapV2RouterContract.methods.getAmountsOut(amounts, path).call();
    return quote;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to get Uniswap V2 router quote');
  }
}

export {
  getWalletBalance,
  getERC20Balance,
  approveERC20,
  swapETHForERC20,
  getUniswapV2RouterQuote,
};
