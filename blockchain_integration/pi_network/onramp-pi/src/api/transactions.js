const express = require('express');
const router = express.Router();
const Transaction = require('../models/Transaction');

router.get('/', async (req, res) => {
  try {
    const transactions = await Transaction.find().sort({ createdAt: -1 });
    res.json(transactions);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to retrieve transactions' });
  }
});

router.post('/', async (req, res) => {
  const { userId, walletAddress, amount, currency, status } = req.body;
  try {
    const transaction = new Transaction({
      userId,
      walletAddress,
      amount,
      currency,
      status,
    });
    await transaction.save();
    res.json(transaction);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create transaction' });
  }
});

module.exports = router;
