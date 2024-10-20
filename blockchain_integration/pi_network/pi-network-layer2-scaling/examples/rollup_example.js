// rollup_example.js
const { ethers } = require("hardhat");
const networkConfig = require("./network_config.json");

async function main() {
    const [owner] = await ethers.getSigners();
    
    // Load the Rollup and RollupManager contracts
    const rollupManagerAddress = networkConfig.networks.rinkeby.rollupManagerAddress;
    const rollupAddress = networkConfig.networks.rinkeby.rollupAddress;

    const Rollup = await ethers.getContractFactory("Rollup");
    const rollup = await Rollup.attach(rollupAddress);

    const RollupManager = await ethers.getContractFactory("RollupManager");
    const rollupManager = await RollupManager.attach(rollupManagerAddress);

    // Create a new batch
    const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
    console.log("Creating a new batch with state root:", stateRoot);
    
    const tx = await rollup.connect(owner).createBatch(stateRoot);
    await tx.wait();
    console.log("Batch created successfully!");

    // Validate the batch
    console.log("Validating the batch...");
    const validatorAddress = await rollupManager.validator();
    const validateTx = await rollup.connect(validatorAddress).validateTransaction(stateRoot, owner.address);
    await validateTx.wait();
    console.log("Batch validated successfully!");

    // Check if the transaction is valid
    const isValid = await rollup.isTransactionValid(stateRoot, owner.address);
    console.log("Is the transaction valid?", isValid);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
