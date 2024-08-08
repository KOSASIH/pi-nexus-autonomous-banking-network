const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Load contract addresses from file
const contractAddresses = require('../contractAddresses.json');

// Set up Binance Smart Chain provider
const provider = new ethers.providers.JsonRpcProvider('https://bsc-dataseed.binance.org/api/v1/ws/bsc/main');

// Set up contract instances
const escrowContract = new ethers.Contract(contractAddresses.Escrow, require('../artifacts/contracts/Escrow.sol/Escrow.json').abi, provider);
const lendingContract = new ethers.Contract(contractAddresses.Lending, require('../artifacts/contracts/Lending.sol/Lending.json').abi, provider);
const tokenizedAssetsContract = new ethers.Contract(contractAddresses.TokenizedAssets, require('../artifacts/contracts/TokenizedAssets.sol/TokenizedAssets.json').abi, provider);

// Set up user wallet
const userWallet = new ethers.Wallet('0x1234567890abcdef', provider);

// Execute contract functions
async function execute() {
  try {
    // Create escrow account
    const createEscrowAccountTx = await escrowContract.connect(userWallet).createEscrowAccount('0x1234567890abcdef', 100);
    console.log(`Create escrow account transaction hash: ${createEscrowAccountTx.hash}`);

    // Deposit into lending pool
    const depositTx = await lendingContract.connect(userWallet).deposit(0, { value: ethers.utils.parseEther('1.0') });
    console.log(`Deposit transaction hash: ${depositTx.hash}`);

    // Create tokenized asset
    const createTokenizedAssetTx = await tokenizedAssetsContract.connect(userWallet).createTokenizedAsset('https://example.com/asset-uri');
    console.log(`Create tokenized asset transaction hash: ${createTokenizedAssetTx.hash}`);

    // Transfer tokenized asset
    const transferTokenizedAssetTx = await tokenizedAssetsContract.connect(userWallet).transferTokenizedAsset(0, '0x1234567890abcdef');
    console.log(`Transfer tokenized asset transaction hash: ${transferTokenizedAssetTx.hash}`);
  } catch (error) {
    console.error(error);
  }
}

execute();
