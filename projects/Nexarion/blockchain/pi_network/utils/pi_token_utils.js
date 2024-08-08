const Web3 = require('web3');

const piTokenContractAbi = require('../contracts/PiTokenContract.json').abi;
const piTokenContractAddress = '0x...'; // Replace with the deployed Pi Token contract address

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

async function isPiTokenTransferAllowed(from, to, amount) {
  try {
    const allowed = await piTokenContract.methods.allowance(from, to).call();
    return allowed >= amount;
  } catch (error) {
    console.error(error);
    return null;
  }
}

module.exports = {
  getPiTokenBalance,
  getPiTokenTotalSupply,
  isPiTokenTransferAllowed
};
