const TransactionModel = require('../models/transactionModel');

// Create a new transaction
const createTransaction = async (req, res) => {
    try {
        const { receiver, amount, currency } = req.body;
        const transaction = new TransactionModel({
            sender: req.user.id,
            receiver,
            amount,
            currency,
        });
        await transaction.save();
        res.status(201).json({ success: true, transaction });
    } catch (error) {
        res.status(400).json({ success: false, message: error.message });
    }
};

// Get a specific transaction by ID
const getTransaction = async (req, res) => {
    try {
        const transaction = await TransactionModel.findById(req.params.id).populate('sender receiver', 'username');
        if (!transaction) {
            return res.status(404).json({ success: false, message: 'Transaction not found' });
        }
        res.json({ success: true, transaction });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

// Get all transactions for a user
const getUserTransactions = async (req, res) => {
    try {
        const transactions = await TransactionModel.find({
            $or: [{ sender: req.user.id }, { receiver: req.user.id }],
        }).populate('sender receiver', 'username');
        res.json({ success: true, transactions });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    createTransaction,
    getTransaction,
    getUserTransactions,
};
