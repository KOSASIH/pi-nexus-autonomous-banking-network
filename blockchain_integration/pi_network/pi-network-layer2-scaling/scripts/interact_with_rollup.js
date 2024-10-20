// interact_with_rollup.js
const { ethers } = require("hardhat");

async function main() {
    const [owner, operator] = await ethers.getSigners();

    // Assuming the addresses of the deployed contracts are known
    const rollupManagerAddress = "YOUR_ROLLUP_MANAGER_ADDRESS";
    const rollupAddress = "YOUR_ROLLUP_ADDRESS";
    const rollupValidatorAddress = "YOUR_ROLLUP_VALIDATOR_ADDRESS";

    const Rollup = await ethers.getContractAt("Rollup", rollupAddress);
    const RollupManager = await ethers.getContractAt("RollupManager", rollupManagerAddress);
    const RollupValidator = await ethers.getContractAt("RollupValidator", rollupValidatorAddress);

    // Add an operator
    await RollupManager.connect(owner).addOperator(operator.address);
    console.log("Operator added:", operator.address);

    // Create a batch
    const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
    await Rollup.connect(operator).createBatch(stateRoot);
    console.log("Batch created with state root:", stateRoot);

    // Validate a transaction
    await RollupValidator.connect(operator).validateTransaction(stateRoot, operator.address);
    console.log("Transaction validated for operator:", operator.address);
}

// Execute the interaction script
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
