const express = require('express');
const { createTransaction, getTransaction, getUserTransactions } = require('../services/transactionService');
const authMiddleware = require('../middleware/authMiddleware');

const router = express.Router();

// Route to create a new transaction
router.post('/', authMiddleware, createTransaction);

// Route to get a specific transaction by ID
router.get('/:id', authMiddleware, getTransaction);

// Route to get all transactions for the logged-in user
router.get('/', authMiddleware, getUserTransactions);

module.exports = router;
