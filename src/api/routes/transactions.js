const express = require('express');
const router = express.Router();
const blockchainService = require('../services/blockchain');

router.post('/transfer', async (req, res) => {
    const { sender, recipient, amount } = req.body;
    try {
        const transaction = await blockchainService.transfer(sender, recipient, amount);
        res.json(transaction);
    } catch (error) {
        res.status(500).json({ error: 'Transaction failed' });
    }
});

module.exports = router;
