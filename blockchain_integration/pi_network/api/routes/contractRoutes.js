// api/routes/contractRoutes.js

const express = require('express');
const contractController = require('../controllers/contractController');

const router = express.Router();

// Get contract details
router.get('/:contractAddress', contractController.getContractDetails);

// Interact with contract (e.g., call a function)
router.post('/:contractAddress/interact', contractController.interactWithContract);

module.exports = router;
