const Web3 = require('web3');

const piNetworkGovernanceContractAbi = require('../contracts/PiNetworkGovernanceContract.json').abi;
const piNetworkGovernanceContractAddress = '0x...'; // Replace with the deployed Pi Network Governance contract address

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const piNetworkGovernanceContract = new web3.eth.Contract(piNetworkGovernanceContractAbi, piNetworkGovernanceContractAddress);

async function getProposalCount() {
  try {
    const proposalCount = await piNetworkGovernanceContract.methods.getProposalCount().call();
    return proposalCount;
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function getProposalById(proposalId) {
  try {
    const proposal = await piNetworkGovernanceContract.methods.getProposalById(proposalId).call();
    return proposal;
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function getVoteCount(proposalId) {
  try {
    const voteCount = await piNetworkGovernanceContract.methods.getVoteCount(proposalId).call();
    return voteCount;
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function hasVoted(proposalId, userAddress) {
  try {
    const hasVoted = await piNetworkGovernanceContract.methods.hasVoted(proposalId, userAddress).call();
    return hasVoted;
  } catch (error) {
    console.error(error);
    return null;
  }
}

module.exports = {
  getProposalCount,
  getProposalById,
  getVoteCount,
  hasVoted
};
