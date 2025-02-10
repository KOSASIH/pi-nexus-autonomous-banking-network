const express = require('express');
const { getAllUsers, getAllTransactions } = require('../services/adminService');
const authMiddleware = require('../middleware/authMiddleware');

const router = express.Router();

// Route to get all users
router.get('/users', authMiddleware, getAllUsers);

// Route to get all transactions
router.get('/transactions', authMiddleware, getAllTransactions);

module.exports = router;
