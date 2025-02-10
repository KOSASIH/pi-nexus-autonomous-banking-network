const ContractModel = require('../models/contractModel');

// Create a new smart contract
const createContract = async (req, res) => {
    try {
        const { address, abi } = req.body;
        const contract = new ContractModel({
            address,
            abi,
            owner: req.user.id,
        });
        await contract.save();
        res.status(201).json({ success: true, contract });
    } catch (error) {
        res.status(400).json({ success: false, message: error.message });
    }
};

// Get all contracts for a user
const getUserContracts = async (req, res) => {
    try {
        const contracts = await ContractModel.find({ owner: req.user.id });
        res.json({ success: true, contracts });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    createContract,
    getUserContracts,
};
