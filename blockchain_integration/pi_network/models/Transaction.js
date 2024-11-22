const mongoose = require('mongoose');

const transactionSchema = new mongoose.Schema({
    sender: {
        type: String,
        required: true,
    },
    receiver: {
        type: String,
        required: true,
    },
    amount: {
        type: Number,
        required: true,
    },
    transactionHash: {
        type: String,
        required: true,
        unique: true,
    },
    status: {
        type: String,
        enum: ['pending', 'completed', 'failed'],
        default: 'pending',
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
    updatedAt: {
        type: Date,
        default: Date.now,
    },
});

// Middleware to update the updatedAt field
transactionSchema.pre('save', function (next) {
    this.updatedAt = Date.now();
    next();
});

const Transaction = mongoose.model('Transaction', transactionSchema);
module.exports = Transaction;
