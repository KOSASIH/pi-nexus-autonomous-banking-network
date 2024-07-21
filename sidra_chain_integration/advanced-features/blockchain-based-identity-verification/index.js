// index.js
const Web3 = require('web3');
const contract = require('./contracts/identity-verification-contract.json');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractInstance = new web3.eth.Contract(contract.abi, contract.networks['5777'].address);

async function verifyUser(userAddress) {
  try {
    await contractInstance.methods.verifyUser(userAddress).send({ from: '0xYOUR_OWNER_ADDRESS' });
    console.log(`User ${userAddress} has been verified`);
  } catch (error) {
    console.error(`Error verifying user: ${error}`);
  }
}

async function isUserVerified(userAddress) {
  try {
    const isVerified = await contractInstance.methods.isUserVerified(userAddress).call();
    console.log(`User ${userAddress} is ${isVerified ? 'verified' : 'not verified'}`);
  } catch (error) {
    console.error(`Error checking user verification: ${error}`);
  }
}

// Example usage:
verifyUser('0xUSER_ADDRESS');
isUserVerified('0xUSER_ADDRESS');
