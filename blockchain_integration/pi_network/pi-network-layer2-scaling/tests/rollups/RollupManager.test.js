// RollupManager.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("RollupManager Contract", function () {
    let RollupManager;
    let rollupManager;
    let owner;
    let operator;

    beforeEach(async function () {
        [owner, operator] = await ethers.getSigners();
        RollupManager = await ethers.getContractFactory("RollupManager");
        rollupManager = await RollupManager.deploy();
        await rollupManager.deployed();
    });

    it("should add an operator successfully", async function () {
        await rollupManager.connect(owner).addOperator(operator.address);
        const isOperator = await rollupManager.isOperator(operator.address);
        expect(isOperator).to.be.true;
    });

    it("should revert when a non-owner tries to add an operator", async function () {
        await expect(rollupManager.connect(operator).addOperator(operator.address)).to.be.revertedWith("Ownable: caller is not the owner");
    });

    it("should remove an operator successfully", async function () {
        await rollupManager.connect(owner).addOperator(operator.address);
        await rollupManager.connect(owner).removeOperator(operator.address);
        const isOperator = await rollupManager.isOperator(operator.address);
        expect(isOperator).to.be.false;
    });

    it("should revert when removing a non-existent operator", async function () {
        await expect(rollupManager.connect(owner).removeOperator(operator.address)).to.be.revertedWith("Operator does not exist");
    });
});
