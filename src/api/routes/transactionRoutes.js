const express = require('express');
const { createTransaction, getTransaction, getAllTransactions } = require('../services/transactionService');

const router = express.Router();

// Route to create a new transaction
router.post('/', createTransaction);

// Route to get a specific transaction by ID
router.get('/:id', getTransaction);

// Route to get all transactions
router.get('/', getAllTransactions);

module.exports = router;
