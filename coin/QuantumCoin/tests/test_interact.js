// tests/test_interact.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Interaction Tests", function () {
    let MultiSigWallet, multiSigWallet;
    let owner1, owner2, owner3;

    before(async function () {
        [owner1, owner2, owner3] = await ethers.getSigners();
        const owners = [owner1.address, owner2.address, owner3.address];
        const requiredConfirmations = 2;

        MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
        multiSigWallet = await MultiSigWallet.deploy(owners, requiredConfirmations);
        await multiSigWallet.deployed();
    });

    it("should submit a transaction", async function () {
        const tx = await multiSigWallet.submitTransaction(owner1.address, ethers.utils.parseEther("1.0"), "0x");
        await tx.wait();

        const txCount = await multiSigWallet.getTransactionCount();
        expect(txCount).to.equal(1);
    });

    it("should confirm a transaction", async function () {
        await multiSigWallet.connect(owner1).confirmTransaction(0);
        const transaction = await multiSigWallet.transactions(0);
        expect(transaction.confirmations).to.equal(1);
    });

    it("should execute a transaction after enough confirmations", async function () {
        await multiSigWallet.connect(owner2).confirmTransaction(0);
        const transaction = await multiSigWallet.transactions(0);
        expect(transaction.executed).to.be.true;
    });
});
