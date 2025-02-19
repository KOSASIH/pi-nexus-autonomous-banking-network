const GovernanceModel = require('../models/governanceModel');

// Create a new governance proposal
const createProposal = async (req, res) => {
    try {
        const { title, description } = req.body;
        const proposal = new GovernanceModel({
            title,
            description,
            createdBy: req.user.id,
        });
        await proposal.save();
        res.status(201).json({ success: true, proposal });
    } catch (error) {
        res.status(400).json({ success: false, message: error.message });
    }
};

// Vote on a proposal
const voteOnProposal = async (req, res) => {
    try {
        const { proposalId, vote } = req.body;
        const proposal = await GovernanceModel.findById(proposalId);
        if (!proposal) {
            return res.status(404).json({ success: false, message: 'Proposal not found' });
        }

        // Check if the user has already voted
        const existingVote = proposal.votes.find(v => v.userId.toString() === req.user.id);
        if (existingVote) {
            return res.status(400).json({ success: false, message: 'User  has already voted' });
        }

        proposal.votes.push({ userId: req.user.id, vote });
        await proposal.save();
        res.json({ success: true, proposal });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

// Get all proposals
const getProposals = async (req, res) => {
    try {
        const proposals = await GovernanceModel.find().populate('createdBy', 'username');
        res.json({ success: true, proposals });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    createProposal,
    voteOnProposal,
    getProposals,
};
