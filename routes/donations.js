// routes/donations.js
const express = require('express');
const router = express.Router();
const Donation = require('../models/Donation');
const Transaction = require('../models/Transaction');

router.post('/donate', async (req, res) => {
  const transaction = await Transaction.findById(req.body.transactionId);
  if (!transaction) {
    res.status(404).send({ message: 'Transaction not found' });
    return;
  }
  const donation = new Donation({
    transaction: transaction,
    amount: req.body.amount,
    cause: req.body.cause,
  });
  await donation.save();
  res.send({ message: 'Donation successful' });
});

module.exports = router;
