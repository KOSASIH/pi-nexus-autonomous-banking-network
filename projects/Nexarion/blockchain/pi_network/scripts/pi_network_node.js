const Web3 = require('web3');
const abi = require('../contracts/PiNetworkContract.json').abi;
const contractAddress = '0x...'; // Replace with the deployed Pi Network contract address

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const piNetworkContract = new web3.eth.Contract(abi, contractAddress);

async function addNode(nodeAddress) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkContract.methods.addNode(nodeAddress).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`Node added: ${nodeAddress}`);
  } catch (error) {
    console.error(error);
  }
}

async function removeNode(nodeAddress) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkContract.methods.removeNode(nodeAddress).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`Node removed: ${nodeAddress}`);
  } catch (error) {
    console.error(error);
  }
}

async function updatePiBalance(userAddress, newBalance) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piNetworkContract.methods.updatePiBalance(userAddress, newBalance).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`PI balance updated: ${userAddress} - ${newBalance}`);
  } catch (error) {
    console.error(error);
  }
}

module.exports = {
  addNode,
  removeNode,
  updatePiBalance
};
