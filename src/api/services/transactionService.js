const TransactionModel = require('../models/transactionModel');

// Create a new transaction
const createTransaction = async (req, res) => {
    try {
        const transaction = new TransactionModel(req.body);
        await transaction.save();
        res.status(201).json({ success: true, transaction });
    } catch (error) {
        res.status(400).json({ success: false, message: error.message });
    }
};

// Get a specific transaction by ID
const getTransaction = async (req, res) => {
    try {
        const transaction = await TransactionModel.findById(req.params.id);
        if (!transaction) {
            return res.status(404).json({ success: false, message: 'Transaction not found' });
        }
        res.json({ success: true, transaction });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

// Get all transactions
const getAllTransactions = async (req, res) => {
    try {
        const transactions = await TransactionModel.find();
        res.json({ success: true, transactions });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    createTransaction,
    getTransaction,
    getAllTransactions,
};
