// scripts/interact.js
const { ethers, run } = require("hardhat");
require("dotenv").config();

async function main() {
    const [user] = await ethers.getSigners();

    // Replace with the actual deployed contract addresses
    const escrowContractAddress = process.env.ESCROW_CONTRACT_ADDRESS;
    const savingsPlanContractAddress = process.env.SAVINGS_PLAN_CONTRACT_ADDRESS;
    const investmentPortfolioAddress = process.env.INVESTMENT_PORTFOLIO_ADDRESS;

    const EscrowContract = await ethers.getContractAt("EscrowContract", escrowContractAddress);
    const SavingsPlanContract = await ethers.getContractAt("SavingsPlanContract", savingsPlanContractAddress);
    const InvestmentPortfolio = await ethers.getContractAt("InvestmentPortfolio", investmentPortfolioAddress);

    console.log("Interacting with contracts as:", user.address);

    // Example: Create an escrow
    const sellerAddress = "0xSellerAddress"; // Replace with actual seller address
    const arbiterAddress = "0xArbiterAddress"; // Replace with actual arbiter address
    const escrowAmount = ethers.utils.parseEther("1.0"); // Amount in ETH

    try {
        const escrowTx = await EscrowContract.createEscrow(sellerAddress, arbiterAddress, { value: escrowAmount });
        await escrowTx.wait();
        console.log("Escrow created successfully!");
    } catch (error) {
        console.error("Error creating escrow:", error);
    }

    // Example: Create a savings plan
    const savingsAmount = ethers.utils.parseEther("1.0"); // Amount in ETH
    const interestRate = 500; // 5% interest rate
    const duration = 30 * 24 * 60 * 60; // 30 days in seconds

    try {
        const savingsTx = await SavingsPlanContract.createSavingsPlan(savingsAmount, interestRate, duration);
        await savingsTx.wait();
        console.log("Savings plan created successfully!");
    } catch (error) {
        console.error("Error creating savings plan:", error);
    }

    // Example: Invest in the portfolio
    const investmentAmount = ethers.utils.parseEther("1.0"); // Amount in tokens
    const tokenAddress = process.env.TOKEN_ADDRESS; // Replace with actual token address

    try {
        const investTx = await InvestmentPortfolio.invest(tokenAddress, investmentAmount);
        await investTx.wait();
        console.log("Investment made successfully!");
    } catch (error) {
        console.error("Error making investment:", error);
    }

    // Example: Withdraw investment
    const withdrawAmount = ethers.utils.parseEther("0.5"); // Amount to withdraw

    try {
        const withdrawTx = await InvestmentPortfolio.withdrawInvestment(tokenAddress, withdrawAmount);
        await withdrawTx.wait();
        console.log("Withdrawal successful!");
    } catch (error) {
        console.error("Error withdrawing investment:", error);
    }

    // Example: Close portfolio
    try {
        const closeTx = await InvestmentPortfolio.closePortfolio();
        await closeTx.wait();
        console.log("Portfolio closed successfully!");
    } catch (error) {
        console.error("Error closing portfolio:", error);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Interaction failed:", error);
        process.exit(1);
    });
