const express = require('express');
const { getAllUsers, getAllTransactions } = require('../services/adminService');
const authMiddleware = require('../middleware/authMiddleware');
const authorizationMiddleware = require('../middleware/authorizationMiddleware');

const router = express.Router();

// Route to get all users
router.get('/users', authMiddleware, authorizationMiddleware(['admin']), getAllUsers);

// Route to get all transactions
router.get('/transactions', authMiddleware, authorizationMiddleware(['admin']), getAllTransactions);

module.exports = router;
