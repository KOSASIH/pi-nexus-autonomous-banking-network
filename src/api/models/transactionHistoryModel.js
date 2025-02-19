const mongoose = require('mongoose');

const transactionHistorySchema = new mongoose.Schema({
    transactionId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Transaction',
        required: true,
    },
    action: {
        type: String,
        enum: ['created', 'updated', 'deleted'],
        required: true,
    },
    timestamp: {
        type: Date,
        default: Date.now,
    },
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
});

const TransactionHistoryModel = mongoose.model('TransactionHistory', transactionHistorySchema);
module.exports = TransactionHistoryModel;
