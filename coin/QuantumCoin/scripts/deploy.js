// scripts/deploy.js
const { ethers } = require("hardhat");

async function main() {
    // Deploy MultiSigWallet
    const MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
    const owners = ["0xYourAddress1", "0xYourAddress2", "0xYourAddress3"]; // Replace with actual addresses
    const requiredConfirmations = 2; // Number of confirmations required
    const multiSigWallet = await MultiSigWallet.deploy(owners, requiredConfirmations);
    await multiSigWallet.deployed();
    console.log("MultiSigWallet deployed to:", multiSigWallet.address);

    // Deploy GovernanceContract
    const GovernanceContract = await ethers.getContractFactory("GovernanceContract");
    const governanceContract = await GovernanceContract.deploy(multiSigWallet.address);
    await governanceContract.deployed();
    console.log("GovernanceContract deployed to:", governanceContract.address);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
