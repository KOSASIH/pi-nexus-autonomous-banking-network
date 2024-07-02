import Web3 from 'web3';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractAddress = '0x...';
const contractABI = [...];

const contract = new web3.eth.Contract(contractABI, contractAddress);

export default contract;
