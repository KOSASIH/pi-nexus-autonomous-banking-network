const express = require('express');
const app = express();
const web3 = require('web3');

app.use(express.json());

const contractAddress = '0x...';
const abi = [...]; // Your contract ABI

const contract = new web3.eth.Contract(abi, contractAddress);

app.post('/transfer', async (req, res) => {
    const { from, to, value } = req.body;

    try {
        await contract.methods.transfer(to, value).send({ from });
        res.json({ message: 'Transfer successful' });
    } catch (error) {
        res.status(500).json({ message: 'Transfer failed' });
    }
});

app.listen(3000, () => {
    console.log('Server listening on port 3000');
});
