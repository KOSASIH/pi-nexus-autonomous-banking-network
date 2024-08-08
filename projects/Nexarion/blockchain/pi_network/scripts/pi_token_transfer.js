const Web3 = require('web3');
const abi = require('../contracts/PiTokenContract.json').abi;
const contractAddress = '0x...'; // Replace with the deployed Pi Token contract address

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const piTokenContract = new web3.eth.Contract(abi, contractAddress);

async function transferPiTokens(recipient, amount) {
  try {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
      from: '0x...', // Replace with the sender's Ethereum address
      to: contractAddress,
      value: '0',
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
      data: piTokenContract.methods.transfer(recipient, amount).encodeABI()
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's Ethereum private key
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log(`PI tokens transferred: ${recipient} - ${amount}`);
  } catch (error) {
    console.error(error);
  }
}

module.exports = {
  transferPiTokens
};
