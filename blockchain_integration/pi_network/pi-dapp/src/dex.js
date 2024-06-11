import express from 'express';
import Web3 from 'web3';

const dex = express.Router();
const web3 = new Web3(new Web3.providers.HttpProvider('https://pi-network.io/rpc'));

dex.get('/orders', async (req, res) => {
    const orders = await web3.eth.getOrders();
    res.json({ orders });
});

dex.post('/placeOrder', async(req, res) => {
    const order = req.body.order;
    const tx = await web3.eth.placeOrder(order);
    res.json({ txHash: tx.transactionHash });
});

export default dex;
