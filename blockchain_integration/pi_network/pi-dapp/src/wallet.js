import express from 'express';
import Web3 from 'web3';

const wallet = express.Router();
const web3 = new Web3(new Web3.providers.HttpProvider('https://pi-network.io/rpc'));

wallet.get('/balance', async (req, res) => {
    const address = req.query.address;
    const balance = await web3.eth.getBalance(address);
    res.json({ balance });
});

wallet.post('/send', async (req, res) => {
    const from = req.body.from;
    const to = req.body.to;
    const amount = req.body.amount;
    const tx = await web3.eth.sendTransaction({ from, to, value: amount });
    res.json({ txHash: tx.transactionHash });
});

export default wallet;
