const path = require('path');
const fs = require('fs');
const Web3 = require('web3');
const TruffleContract = require('truffle-contract');
const HDWalletProvider = require('truffle-hdwallet-provider');

// Set up the Web3 provider
const provider = new HDWalletProvider(
  'YOUR_MNEMONIC_PHRASE',
  'https://mainnet.infura.io/v3/YOUR_PROJECT_ID'
);

// Set up the Web3 instance
const web3 = new Web3(provider);

// Set up the contract artifacts
const HealthRecord = require('../build/contracts/HealthRecord.json');
const MedicalBilling = require('../build/contracts/MedicalBilling.json');

// Set up the contract instances
const healthRecordContract = TruffleContract(HealthRecord);
const medicalBillingContract = TruffleContract(MedicalBilling);

// Set up the deployer
const deployer = async () => {
  // Deploy the HealthRecord contract
  const healthRecordInstance = await healthRecordContract.new({
    from: '0x...YOUR_ADDRESS...',
    gas: 5000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });
  console.log(`HealthRecord contract deployed at address: ${healthRecordInstance.address}`);

  // Deploy the MedicalBilling contract
  const medicalBillingInstance = await medicalBillingContract.new({
    from: '0x...YOUR_ADDRESS...',
    gas: 5000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });
  console.log(`MedicalBilling contract deployed at address: ${medicalBillingInstance.address}`);
};

// Run the deployer
deployer().catch((error) => {
  console.error(error);
});
