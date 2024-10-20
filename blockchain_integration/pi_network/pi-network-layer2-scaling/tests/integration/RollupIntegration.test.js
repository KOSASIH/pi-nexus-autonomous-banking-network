// RollupIntegration.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Rollup Integration Tests", function () {
    let Rollup;
    let rollup;
    let RollupManager;
    let rollupManager;
    let RollupValidator;
    let rollupValidator;
    let owner;
    let operator;

    beforeEach(async function () {
        [owner, operator] = await ethers.getSigners();

        RollupManager = await ethers.getContractFactory("RollupManager");
        rollupManager = await RollupManager.deploy();
        await rollupManager.deployed();

        Rollup = await ethers.getContractFactory("Rollup");
        rollup = await Rollup.deploy(rollupManager.address);
        await rollup.deployed();

        RollupValidator = await ethers.getContractFactory("RollupValidator");
        rollupValidator = await RollupValidator.deploy(rollup.address);
        await rollupValidator.deployed();
    });

    it("should create and validate a batch successfully", async function () {
        const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
        
        // Create a batch
        await rollup.connect(operator).createBatch(stateRoot);
        
        // Validate the batch
        await rollupValidator.connect(operator).validateTransaction(stateRoot, operator.address);
        
        const isValid = await rollup.isTransactionValid(stateRoot, operator.address);
        expect(isValid).to.be.true;
    });

    it("should revert when validating a non-existent batch", async function () {
        const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Non-existent State Root"));
        await expect(rollupValidator.connect(operator).validateTransaction(stateRoot, operator.address)).to.be.revertedWith("Batch does not exist");
    });
});
