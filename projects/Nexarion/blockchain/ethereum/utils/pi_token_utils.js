const Web3 = require('web3');
const piTokenContractAddress = require('../contract_addresses.json').piTokenContractAddress;
const piTokenContractAbi = require('../contracts/PiTokenContract.sol').abi;

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const piTokenContract = new web3.eth.Contract(piTokenContractAbi, piTokenContractAddress);

async function getPiTokenBalance(userAddress) {
  try {
    const balance = await piTokenContract.methods.balanceOf(userAddress).call();
    return balance;
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function getPiTokenTotalSupply() {
  try {
    const totalSupply = await piTokenContract.methods.totalSupply().call();
    return totalSupply;
  } catch (error) {
    console.error(error);
    return null;
  }
}

module.exports = {
  getPiTokenBalance,
  getPiTokenTotalSupply,
};
