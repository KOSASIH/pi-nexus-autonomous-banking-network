// scripts/governance.js
const { ethers } = require("hardhat");

async function main() {
    const governanceContractAddress = "0xYourGovernanceContractAddress"; // Replace with actual address

    const GovernanceContract = await ethers.getContractAt("GovernanceContract", governanceContractAddress);

    // Example: Create a proposal
    const proposalTx = await GovernanceContract.createProposal("Increase the supply of QuantumCoin", 7 * 24 * 60 *  60); // Proposal duration in seconds
    await proposalTx.wait();
    console.log("Proposal created!");

    // Example: Vote on a proposal
    const proposalId = 0; // Replace with the actual proposal ID
    const voteTx = await GovernanceContract.vote(proposalId, true); // true for 'in favor'
    await voteTx.wait();
    console.log("Vote cast!");

    // Example: Execute a proposal
    const executeTx = await GovernanceContract.executeProposal(proposalId);
    await executeTx.wait();
    console.log("Proposal executed!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
