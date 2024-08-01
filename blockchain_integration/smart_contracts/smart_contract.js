// blockchain_integration/smart_contracts/smart_contract.js
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractAddress = '0x...';
const contractABI = [...];

const contract = new web3.eth.Contract(contractABI, contractAddress);

module.exports = contract;
