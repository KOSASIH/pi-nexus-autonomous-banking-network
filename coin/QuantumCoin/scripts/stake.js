// scripts/stake.js
const { ethers } = require("hardhat");

async function main() {
    const quantumCoinAddress = "0xYourQuantumCoinAddress"; // Replace with actual address
    const stakingContractAddress = "0xYourStakingContractAddress"; // Replace with actual address

    const QuantumCoin = await ethers.getContractAt("IERC20", quantumCoinAddress);
    const StakingContract = await ethers.getContractAt("StakingContract", stakingContractAddress);

    const amountToStake = ethers.utils.parseEther("10.0"); // Amount to stake

    // Approve the staking contract to spend tokens
    const approveTx = await QuantumCoin.approve(stakingContractAddress, amountToStake);
    await approveTx.wait();
    console.log("Tokens approved for staking!");

    // Stake the tokens
    const stakeTx = await StakingContract.stake(amountToStake);
    await stakeTx.wait();
    console.log("Tokens staked!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
