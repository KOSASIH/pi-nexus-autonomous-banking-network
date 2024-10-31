// scripts/deploy.js
const { ethers, upgrades } = require("hardhat");
require("dotenv").config();

async function main() {
    // Get the deployer account
    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    // Check the balance of the deployer
    const balance = await deployer.getBalance();
    console.log("Deployer balance:", ethers.utils.formatEther(balance), "ETH");

    // Deploy EscrowContract
    const EscrowContract = await ethers.getContractFactory("EscrowContract");
    const escrowContract = await upgrades.deployProxy(EscrowContract, [], { initializer: 'initialize' });
    await escrowContract.deployed();
    console.log("EscrowContract deployed to:", escrowContract.address);

    // Deploy SavingsPlanContract
    const SavingsPlanContract = await ethers.getContractFactory("SavingsPlanContract");
    const savingsPlanContract = await upgrades.deployProxy(SavingsPlanContract, [], { initializer: 'initialize' });
    await savingsPlanContract.deployed();
    console.log("SavingsPlanContract deployed to:", savingsPlanContract.address);

    // Deploy InvestmentPortfolio
    const InvestmentPortfolio = await ethers.getContractFactory("InvestmentPortfolio");
    const investmentPortfolio = await upgrades.deployProxy(InvestmentPortfolio, [process.env.TOKEN_ADDRESS], { initializer: 'initialize' });
    await investmentPortfolio.deployed();
    console.log("InvestmentPortfolio deployed to:", investmentPortfolio.address);

    // Verify contracts on Etherscan (optional)
    if (process.env.NETWORK === "mainnet" || process.env.NETWORK === "testnet") {
        await verifyContract(escrowContract.address);
        await verifyContract(savingsPlanContract.address);
        await verifyContract(investmentPortfolio.address);
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
        console.error("Deployment failed:", error);
        process.exit(1);
    });
