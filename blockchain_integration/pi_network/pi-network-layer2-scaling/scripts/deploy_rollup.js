// deploy_rollup.js
const { ethers } = require("hardhat");

async function main() {
    // Deploy RollupManager
    const RollupManager = await ethers.getContractFactory("RollupManager");
    const rollupManager = await RollupManager.deploy();
    await rollupManager.deployed();
    console.log("RollupManager deployed to:", rollupManager.address);

    // Deploy Rollup
    const Rollup = await ethers.getContractFactory("Rollup");
    const rollup = await Rollup.deploy(rollupManager.address);
    await rollup.deployed();
    console.log("Rollup deployed to:", rollup.address);

    // Deploy RollupValidator
    const RollupValidator = await ethers.getContractFactory("RollupValidator");
    const rollupValidator = await RollupValidator.deploy(rollup.address);
    await rollupValidator.deployed();
    console.log("RollupValidator deployed to:", rollupValidator.address);
}

// Execute the deployment script
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
