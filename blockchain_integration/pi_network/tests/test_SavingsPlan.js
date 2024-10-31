// tests/test_SavingsPlan.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("SavingsPlan", function () {
    let SavingsPlan;
    let savingsPlan;
    let owner;
    let addr1;

    beforeEach(async function () {
        SavingsPlan = await ethers.getContractFactory("SavingsPlan");
        [owner, addr1] = await ethers.getSigners();
        savingsPlan = await SavingsPlan.deploy();
        await savingsPlan.deployed();
    });

    it("Should create a savings plan", async function () {
        const savingsAmount = ethers.utils.parseEther("1.0");
        const interestRate = 500; // 5%
        const duration = 30 * 24 * 60 * 60; // 30 days

        await savingsPlan.createSavingsPlan(savingsAmount, interestRate, duration);
        const plan = await savingsPlan.getPlan(0);
        expect(plan.amount).to.equal(savingsAmount);
        expect(plan.interestRate).to.equal(interestRate);
    });

    it("Should allow withdrawal after maturity", async function () {
        const savingsAmount = ethers.utils.parseEther("1.0");
        const interestRate = 500; // 5%
        const duration = 1; // 1 second for testing

        await savingsPlan.createSavingsPlan(savingsAmount, interestRate, duration);
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for 2 seconds

        await savingsPlan.withdraw(0);
        expect(await savingsPlan.getBalance(owner.address)).to.be.above(0);
    });

    it("Should not allow withdrawal before maturity", async function () {
        const savingsAmount = ethers.utils.parseEther("1.0");
        const interestRate = 500; // 5%
        const duration = 5; // 5 seconds for testing

        await savingsPlan.createSavingsPlan(savingsAmount, interestRate, duration);
        await expect(savingsPlan.withdraw(0)).to.be.revertedWith("Savings plan not matured yet");
    });
});
