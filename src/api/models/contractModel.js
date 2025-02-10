const mongoose = require('mongoose');

const contractSchema = new mongoose.Schema({
    address: {
        type: String,
        required: true,
        unique: true,
    },
    abi: {
        type: Object,
        required: true,
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
    owner: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
});

const ContractModel = mongoose.model('Contract', contractSchema);
module.exports = ContractModel;
