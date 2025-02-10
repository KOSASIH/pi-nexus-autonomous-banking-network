const mongoose = require('mongoose');

const voteSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User ',
        required: true,
    },
    vote: {
        type: String,
        enum: ['yes', 'no'],
        required: true,
    },
}, { timestamps: true });

const governanceSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
    },
    description: {
        type: String,
        required: true,
    },
    votes: [voteSchema],
    status: {
        type: String,
        enum: ['pending', 'approved', 'rejected'],
        default: 'pending',
    },
    createdBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User ',
        required: true,
    },
}, { timestamps: true });

const GovernanceModel = mongoose.model('Governance', governanceSchema);
module.exports = GovernanceModel;
