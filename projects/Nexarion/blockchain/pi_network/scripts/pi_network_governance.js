const Web3 = require('web3');
const abi = require('../contracts/PiNetworkGovernanceContract.json').abi;
const contractAddress = '0x...'; // Replace with the deployed Pi Network Governance contract address

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const piNetworkGovernanceContract = new web3.eth.Contract(abi, contractAddress);

async function createProposal(description) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkGovernanceContract.methods.createProposal(description).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`Proposal created: ${description}`);
  } catch (error) {
    console.error(error);
  }
}

async function voteOnProposal(proposalId, vote) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkGovernanceContract.methods.voteOnProposal(proposalId, vote).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`Voted on proposal: ${proposalId} - ${vote}`);
  } catch (error) {
    console.error(error);
  }
}

async function executeProposal(proposalId) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkGovernanceContract.methods.executeProposal(proposalId).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`Proposal executed: ${proposalId}`);
  } catch (error) {
    console.error(error);
  }
}

module.exports = {
  createProposal,
  voteOnProposal,
  executeProposal
};
