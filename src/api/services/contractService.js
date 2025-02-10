const Web3 = require('web3');
const { CONTRACT_ABI, CONTRACT_BYTECODE } = require('../config/blockchainConfig');
const web3 = new Web3(new Web3.providers.HttpProvider(process.env.BLOCKCHAIN_NODE_URL));

// Deploy a new smart contract
const deployContract = async (req, res) => {
    try {
        const { from, gas } = req.body;
        const contract = new web3.eth.Contract(CONTRACT_ABI);
        const deployTx = contract.deploy({ data: CONTRACT_BYTECODE });

        const gasEstimate = await deployTx.estimateGas();
        const result = await deployTx.send({ from, gas: gasEstimate });

        res.status(201).json({ success: true, contractAddress: result.options.address });
    } catch (error) {
        res.status(400).json({ success: false, message: error.message });
    }
};

// Get details of a specific contract
const getContractDetails = async (req, res) => {
    try {
        const contract = new web3.eth.Contract(CONTRACT_ABI, req.params.contractAddress);
        const details = await contract.methods.getDetails().call();
        res.json({ success: true, details });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
};

module.exports = {
    deployContract,
    getContractDetails,
};
