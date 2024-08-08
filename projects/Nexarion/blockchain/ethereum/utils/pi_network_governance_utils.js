const Web3 = require('web3');
const piNetworkGovernanceContractAddress = require('../contract_addresses.json').piNetworkGovernanceContractAddress;
const piNetworkGovernanceContractAbi = require('../contracts/PiNetworkGovernanceContract.sol').abi;

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

async function voteOnProposal(proposalId, vote) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...' /* your Ethereum address */,
      to: piNetworkGovernanceContractAddress,
      data: piNetworkGovernanceContract.methods.voteOnProposal(proposalId, vote).encodeABI(),
      gas: '20000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...' /* your Ethereum private key */);
    await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  } catch (error) {
    console.error(error);
  }
}

module.exports = {
  getProposalCount,
  getProposalById,
  voteOnProposal,
};
