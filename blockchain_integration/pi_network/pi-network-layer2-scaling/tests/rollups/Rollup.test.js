// Rollup.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Rollup Contract", function () {
    let Rollup;
    let rollup;
    let RollupManager;
    let rollupManager;
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
    });

    it("should create a batch successfully", async function () {
        const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
        await rollup.connect(operator).createBatch(stateRoot);
        
        const batch = await rollup.getBatch(0);
        expect(batch.stateRoot).to.equal(stateRoot);
        expect(batch.index).to.equal(0);
    });

    it("should validate a transaction successfully", async function () {
        const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
        await rollup.connect(operator).createBatch(stateRoot);
        
        await rollup.connect(operator).validateTransaction(stateRoot, operator.address);
        const isValid = await rollup.isTransactionValid(stateRoot, operator.address);
        expect(isValid).to.be.true;
    });

    it("should revert when creating a batch with the same state root", async function () {
        const stateRoot = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("Sample State Root"));
        await rollup.connect(operator).createBatch(stateRoot);
        
        await expect(rollup.connect(operator).createBatch(stateRoot)).to.be.revertedWith("Batch already exists");
    });
});
