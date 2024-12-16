// ProposalManagement.js

const Web3 = require('web3');
const GovernanceVotingSystemABI = require('./GovernanceVotingSystemABI.json'); // ABI of the GovernanceVotingSystem contract

class ProposalManagement {
    constructor(contractAddress, provider) {
        this.web3 = new Web3(provider);
        this.contract = new this.web3.eth.Contract(GovernanceVotingSystemABI, contractAddress);
    }

    // Create a new proposal
    async createProposal(description, fromAddress) {
        const tx = await this.contract.methods.createProposal(description).send({ from: fromAddress });
        console.log('Proposal created:', tx.events.ProposalCreated.returnValues);
    }

    // Vote on a proposal
    async vote(proposalId, fromAddress) {
        const tx = await this.contract.methods.vote(proposalId).send({ from: fromAddress });
        console.log('Voted on proposal:', tx.events.Voted.returnValues);
    }

    // Execute a proposal
    async executeProposal(proposalId, fromAddress) {
        const tx = await this.contract.methods.executeProposal(proposalId).send({ from: fromAddress });
        console.log('Executed proposal:', tx.events.ProposalExecuted.returnValues);
    }

    // Get proposal details
    async getProposal(proposalId) {
        const proposal = await this.contract.methods.proposals(proposalId).call();
        console.log('Proposal details:', proposal);
        return proposal;
    }
}

// Example usage
const provider = 'https://your.ethereum.node'; // Replace with your Ethereum node provider
const contractAddress = '0xYourContractAddress'; // Replace with your contract address
const proposalManager = new ProposalManagement(contractAddress, provider);

// Replace with actual Ethereum address
const fromAddress = '0xYourEthereumAddress';

// Create a proposal
proposalManager.createProposal('Increase the staking rewards by 5%', fromAddress);

// Vote on a proposal
proposalManager.vote(1, fromAddress);

// Execute a proposal
proposalManager.executeProposal(1, fromAddress);
