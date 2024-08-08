const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Set up Binance Smart Chain provider
const provider = new ethers.providers.JsonRpcProvider('https://bsc-dataseed.binance.org/api/v1/ws/bsc/main');

// Set up contract artifacts
const escrowArtifact = require('../artifacts/contracts/Escrow.sol/Escrow.json');
const lendingArtifact = require('../artifacts/contracts/Lending.sol/Lending.json');
const tokenizedAssetsArtifact = require('../artifacts/contracts/TokenizedAssets.sol/TokenizedAssets.json');

// Set up deployer wallet
const deployerWallet = new ethers.Wallet('0x1234567890abcdef', provider);

// Deploy contracts
async function deployContracts() {
  try {
    // Deploy Escrow contract
    const escrowContract = await deployerWallet.deploy(escrowArtifact, []);
    console.log(`Escrow contract deployed to ${escrowContract.address}`);

    // Deploy Lending contract
    const lendingContract = await deployerWallet.deploy(lendingArtifact, []);
    console.log(`Lending contract deployed to ${lendingContract.address}`);

    // Deploy TokenizedAssets contract
    const tokenizedAssetsContract = await deployerWallet.deploy(tokenizedAssetsArtifact, []);
    console.log(`TokenizedAssets contract deployed to ${tokenizedAssetsContract.address}`);

    // Save contract addresses to file
    const contractAddresses = {
      Escrow: escrowContract.address,
      Lending: lendingContract.address,
      TokenizedAssets: tokenizedAssetsContract.address,
    };
    fs.writeFileSync(path.join(__dirname, '../contractAddresses.json'), JSON.stringify(contractAddresses, null, 2));
  } catch (error) {
    console.error(error);
  }
}

deployContracts();
