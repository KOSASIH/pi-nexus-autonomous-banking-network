const express = require('express');
const bodyParser = require('body-parser');
const Web3 = require('web3');
const { SidraChain } = require('./sidra-chain');

const app = express();
app.use(bodyParser.json());

// Set up Web3 provider
const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

// Set up SidraChain contract instance
const sidraChainContract = new web3.eth.Contract(SidraChain.abi, '0x...SidraChainContractAddress...');

// API endpoint to create a new transaction
app.post('/transactions', async (req, res) => {
  const { from, to, value } = req.body;
  try {
    const txCount = await web3.eth.getTransactionCount(from);
    const tx = {
      from,
      to,
      value,
      gas: '20000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...privateKey...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    res.json({ transactionHash: receipt.transactionHash });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create transaction' });
  }
});

// API endpoint to get user balance
app.get('/balances/:address', async (req, res) => {
  const address = req.params.address;
  try {
    const balance = await sidraChainContract.methods.balanceOf(address).call();
    res.json({ balance: balance.toString() });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to get balance' });
  }
});

// API endpoint to get transaction history
app.get('/transactions/:address', async (req, res) => {
  const address = req.params.address;
  try {
    const transactions = await sidraChainContract.methods.getTransactionHistory(address).call();
    res.json({ transactions: transactions.map(tx => tx.toString()) });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to get transaction history' });
  }
});

// Start the server
const port = 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
