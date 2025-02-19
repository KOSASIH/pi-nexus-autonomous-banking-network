const TransactionHistoryModel = require('../models/transactionHistoryModel');

// Log a transaction action
const logTransactionAction = async (transactionId, action, userId) => {
    const history = new TransactionHistoryModel({ transactionId, action, userId });
    await history.save();
};

module.exports = {
    logTransactionAction,
};
