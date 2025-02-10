const UserModel = require('../models/userModel');
const TransactionModel = require('../models/transactionModel');

// Get all users
const getAllUsers = async (req, res) => {
    try {
        const users = await UserModel.find().select('-password');
        res.json({ success: true, users });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

// Get all transactions
const getAllTransactions = async (req, res) => {
    try {
        const transactions = await TransactionModel.find().populate('sender receiver', 'username');
        res.json({ success: true, transactions });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    getAllUsers,
    getAllTransactions,
};
