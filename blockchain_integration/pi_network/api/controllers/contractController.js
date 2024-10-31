// api/controllers/contractController.js

const { ethers } = require('ethers');

// Replace with your contract ABI and provider
const provider = new ethers.providers.InfuraProvider('mainnet', 'YOUR_INFURA_PROJECT_ID');
const contractABI = [
    // Replace with your contract's ABI
    // Example ABI for a simple contract
    "function getDetails() view returns (string memory, uint256)",
    "function interact(uint256 value) public returns (bool)"
];

const contractController = {
    getContractDetails: async (req, res) => {
        const { contractAddress } = req.params;

        // Create a contract instance
        const contract = new ethers.Contract(contractAddress, contractABI, provider);

        try {
            // Call the contract's getDetails method
            const details = await contract.getDetails();
            const response = {
                name: details[0], // Assuming the first return value is a string (e.g., name)
                value: details[1] // Assuming the second return value is a uint256 (e.g., value)
            };
            res.json({ contractAddress, details: response });
        } catch (error) {
            console.error("Error fetching contract details:", error);
            res.status(500).json({ message: 'Error fetching contract details', error: error.message });
        }
    },

    interactWithContract: async (req, res) => {
        const { contractAddress } = req.params;
        const { value } = req.body; // Value to interact with the contract

        // Replace with your wallet private key (use environment variables in production)
        const wallet = new ethers.Wallet('YOUR_PRIVATE_KEY', provider);
        const contract = new ethers.Contract(contractAddress, contractABI, wallet);

        try {
            // Call the contract's interact method
            const tx = await contract.interact(value);
            await tx.wait(); // Wait for the transaction to be mined

            res.json({ message: 'Interaction successful', transactionHash: tx.hash });
        } catch (error) {
            console.error("Error interacting with contract:", error);
            res.status(500).json({ message: 'Error interacting with contract', error: error.message });
        }
    }
};

module.exports = contractController;
