const express = require('express');
const router = express.Router();
const stellarService = require('./stellarService');

// Route to create a new Stellar account
router.post('/create-account', async (req, res) => {
    try {
        const account = await stellarService.createAccount();
        res.status(201).json({
            publicKey: account.publicKey(),
            secret: account.secret(),
            message: 'Account created successfully. Fund it using the Stellar Testnet Faucet.'
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Route to send a payment on the Stellar network
router.post('/send-payment', async (req, res) => {
    const { sourceSecret, destinationPublicKey, amount } = req.body;

    // Validate input
    if (!sourceSecret || !destinationPublicKey || !amount) {
        return res.status(400).json({ error: 'Source secret, destination public key, and amount are required.' });
    }

    try {
        const result = await stellarService.sendPayment(sourceSecret, destinationPublicKey, amount);
        res.status(200).json({
            success: true,
            transactionId: result.id,
            message: 'Payment sent successfully!'
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Route to get account details
router.get('/account/:publicKey', async (req, res) => {
    const { publicKey } = req.params;

    try {
        const accountDetails = await stellarService.getAccountDetails(publicKey);
        res.status(200).json(accountDetails);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;
