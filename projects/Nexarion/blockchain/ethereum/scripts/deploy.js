const Web3 = require('web3');
const fs = require('fs');

const piNetworkContract = require('../contracts/PiNetworkContract.sol');
const piTokenContract = require('../contracts/PiTokenContract.sol');
const piNetworkGovernanceContract = require('../contracts/PiNetworkGovernanceContract.sol');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

async function deployContracts() {
  try {
    // Deploy Pi Network Contract
    const piNetworkContractInstance = new web3.eth.Contract(piNetworkContract.abi);
    const piNetworkContractTx = piNetworkContractInstance.deploy({
      data: piNetworkContract.bytecode,
    });
    const piNetworkContractReceipt = await web3.eth.sendTransaction(piNetworkContractTx);
    const piNetworkContractAddress = piNetworkContractReceipt.contractAddress;
    console.log(`Pi Network Contract deployed to ${piNetworkContractAddress}`);

    // Deploy Pi Token Contract
    const piTokenContractInstance = new web3.eth.Contract(piTokenContract.abi);
    const piTokenContractTx = piTokenContractInstance.deploy({
      data: piTokenContract.bytecode,
    });
    const piTokenContractReceipt = await web3.eth.sendTransaction(piTokenContractTx);
    const piTokenContractAddress = piTokenContractReceipt.contractAddress;
    console.log(`Pi Token Contract deployed to ${piTokenContractAddress}`);

    // Deploy Pi Network Governance Contract
    const piNetworkGovernanceContractInstance = new web3.eth.Contract(piNetworkGovernanceContract.abi);
    const piNetworkGovernanceContractTx = piNetworkGovernanceContractInstance.deploy({
      data: piNetworkGovernanceContract.bytecode,
    });
    const piNetworkGovernanceContractReceipt = await web3.eth.sendTransaction(piNetworkGovernanceContractTx);
    const piNetworkGovernanceContractAddress = piNetworkGovernanceContractReceipt.contractAddress;
    console.log(`Pi Network Governance Contract deployed to ${piNetworkGovernanceContractAddress}`);

    // Save contract addresses to file
    fs.writeFileSync('contract_addresses.json', JSON.stringify({
      piNetworkContractAddress,
      piTokenContractAddress,
      piNetworkGovernanceContractAddress,
    }));
  } catch (error) {
    console.error(error);
  }
}

deployContracts();
