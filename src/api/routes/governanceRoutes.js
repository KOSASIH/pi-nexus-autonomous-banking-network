const express = require('express');
const { createProposal, voteOnProposal, getProposals } = require('../services/governanceService');
const validateUser  = require('../middleware/validationMiddleware');

const router = express.Router();

// Route to create a new governance proposal
router.post('/', validateUser , createProposal);

// Route to vote on a proposal
router.post('/vote', validateUser , voteOnProposal);

// Route to get all proposals
router.get('/', getProposals);

module.exports = router;
