const express = require('express');
const { deployContract, getContractDetails } = require('../services/contractService');

const router = express.Router();

// Route to deploy a new smart contract
router.post('/deploy', deployContract);

// Route to get details of a specific contract
router.get('/:contractAddress', getContractDetails);

module.exports = router;
