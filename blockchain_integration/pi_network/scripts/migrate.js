// scripts/migrate.js
const { ethers, upgrades } = require("hardhat");
require("dotenv").config();

async function main() {
    // Get the deployer account
    const [deployer] = await ethers.getSigners();
    console.log("Migrating contracts with the account:", deployer.address);

    // Replace with the actual deployed contract addresses
    const oldEscrowContractAddress = process.env.OLD_ESCROW_CONTRACT_ADDRESS;
    const oldSavingsPlanContractAddress = process.env.OLD_SAVINGS_PLAN_CONTRACT_ADDRESS;
    const oldInvestmentPortfolioAddress = process.env.OLD_INVESTMENT_PORTFOLIO_ADDRESS;

    // Upgrade EscrowContract
    const EscrowContract = await ethers.getContractFactory("EscrowContract");
    const upgradedEscrowContract = await upgrades.upgradeProxy(oldEscrowContractAddress, EscrowContract);
    await upgradedEscrowContract.deployed();
    console.log("EscrowContract upgraded to:", upgradedEscrowContract.address);

    // Upgrade SavingsPlanContract
    const SavingsPlanContract = await ethers.getContractFactory("SavingsPlanContract");
    const upgradedSavingsPlanContract = await upgrades.upgradeProxy(oldSavingsPlanContractAddress, SavingsPlanContract);
    await upgradedSavingsPlanContract.deployed();
    console.log("SavingsPlanContract upgraded to:", upgradedSavingsPlanContract.address);

    // Upgrade InvestmentPortfolio
    const InvestmentPortfolio = await ethers.getContractFactory("InvestmentPortfolio");
    const upgradedInvestmentPortfolio = await upgrades.upgradeProxy(oldInvestmentPortfolioAddress, InvestmentPortfolio);
    await upgradedInvestmentPortfolio.deployed();
    console.log("InvestmentPortfolio upgraded to:", upgradedInvestmentPortfolio.address);

    // Verify upgraded contracts on Etherscan (optional)
    if (process.env.NETWORK === "mainnet" || process.env.NETWORK === "testnet") {
        await verifyContract(upgradedEscrowContract.address);
        await verifyContract(upgradedSavingsPlanContract.address);
        await verifyContract(upgradedInvestmentPortfolio.address);
    }
}

async function verifyContract(contractAddress) {
    console.log(`Verifying contract at ${contractAddress}...`);
    try {
        await run("verify:verify", {
            address: contractAddress,
            constructorArguments: [],
        });
        console.log("Contract verified successfully!");
    } catch (error) {
        console.error("Verification failed:", error);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Migration failed:", error);
        process.exit(1);
    });
