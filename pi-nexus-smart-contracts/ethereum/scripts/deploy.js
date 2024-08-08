// Import required libraries
const Web3 = require('web3');
const ethers = require('ethers');
const fs = require('fs');
const path = require('path');

// Set up Web3 provider
const provider = new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID');

// Set up Ethereum wallet
const wallet = new ethers.Wallet('0xYOUR_PRIVATE_KEY');

// Load contract artifacts
const lendingPoolArtifact = require('../artifacts/LendingPool.json');
const tokenizedAssetsArtifact = require('../artifacts/TokenizedAssets.json');

// Deploy LendingPool contract
async function deployLendingPool() {
  const lendingPoolContract = new ethers.ContractFactory(lendingPoolArtifact.abi, lendingPoolArtifact.bytecode, wallet);
  const lendingPoolInstance = await lendingPoolContract.deploy();
  console.log(`LendingPool contract deployed at ${lendingPoolInstance.address}`);

  // Save contract address to file
  fs.writeFileSync(path.join(__dirname, '../contracts/LendingPool.address'), lendingPoolInstance.address);
}

// Deploy TokenizedAssets contract
async function deployTokenizedAssets() {
  const tokenizedAssetsContract = new ethers.ContractFactory(tokenizedAssetsArtifact.abi, tokenizedAssetsArtifact.bytecode, wallet);
  const tokenizedAssetsInstance = await tokenizedAssetsContract.deploy();
  console.log(`TokenizedAssets contract deployed at ${tokenizedAssetsInstance.address}`);

  // Save contract address to file
  fs.writeFileSync(path.join(__dirname, '../contracts/TokenizedAssets.address'), tokenizedAssetsInstance.address);
}

// Deploy both contracts
async function deploy() {
  await deployLendingPool();
  await deployTokenizedAssets();
}

deploy();
