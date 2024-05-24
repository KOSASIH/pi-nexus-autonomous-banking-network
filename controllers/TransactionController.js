// controllers/TransactionController.js
const Transaction = require('../models/Transaction');

const trackTransaction = async (req, res) => {
  // Implement logic to track the transaction
  // For example, using a third-party payment gateway API
  const transaction = new Transaction(req.body);
  try {
    await transaction.save();
    res.status(201).send({ message: 'Transaction tracked successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).send({ message: 'Error tracking transaction' });
  }
};

const getTransactionHistory = async (req, res) => {
  const transactions = await Transaction.find({ user: req.user.id });
  res.send(transactions);
};

module.exports = { trackTransaction, getTransactionHistory };
