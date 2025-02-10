const express = require('express');
const { createContract, getUserContracts } = require('../services/contractService');
const authMiddleware = require('../middleware/authMiddleware');

const router = express.Router();

// Route to create a new smart contract
router.post('/', authMiddleware, createContract);

// Route to get all contracts for the logged-in user
router.get('/', authMiddleware, getUserContracts);

module.exports = router;
