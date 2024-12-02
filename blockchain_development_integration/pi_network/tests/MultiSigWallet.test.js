// MultiSigWallet.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MultiSigWallet Contract", function () {
    let MultiSigWallet;
    let multiSigWallet;
    let owner;
    let addr1;
    let addr2;
    let addr3;

    beforeEach(async function () {
        MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
        [owner, addr1, addr2, addr3] = await ethers.getSigners();
        multiSigWallet = await MultiSigWallet.deploy([owner.address, addr1.address, addr2.address], 2); // 2 confirmations required
        await multiSigWallet.deployed();
    });

    it("Should allow a user to submit a transaction", async function () {
        const txValue = ethers.utils.parseEther("1.0");
        await multiSigWallet.connect(owner).submitTransaction(addr3.address, txValue, "0x");

        const transaction = await multiSigWallet.transactions(0);
        expect(transaction.to).to.equal(addr3.address);
        expect(transaction.value).to.equal(txValue);
        expect(transaction.confirmations).to.equal(0);
    });

    it("Should allow multiple owners to confirm a transaction", async function () {
        const txValue = ethers.utils.parseEther("1.0");
        await multiSigWallet.connect(owner).submitTransaction(addr3.address, txValue, "0x");
        
        await multiSigWallet.connect(owner).confirmTransaction(0);
        await multiSigWallet.connect(addr1).confirmTransaction(0);

        const transaction = await multiSigWallet.transactions(0);
        expect(transaction.confirmations).to.equal(2); // 2 confirmations
    });

    it("Should execute a transaction after sufficient confirmations", async function () {
        const txValue = ethers.utils.parseEther("1.0");
        await multiSigWallet.connect(owner).submitTransaction(addr3.address, txValue, "0x");
        
        await multiSigWallet.connect(owner).confirmTransaction(0);
        await multiSigWallet.connect(addr1).confirmTransaction(0);

        const initialBalance = await ethers.provider.getBalance(addr3.address);
        await multiSigWallet.connect(owner).executeTransaction(0);
        const finalBalance = await ethers.provider.getBalance(addr3.address);

        expect(finalBalance).to.equal(initialBalance.add(txValue)); // addr3's balance should increase
    });

    it("Should revert if trying to execute a transaction without enough confirmations", async function () {
        const txValue = ethers.utils.parseEther("1.0");
        await multiSigWallet.connect(owner).submitTransaction(addr3.address, txValue, "0x");
        
        await expect(multiSigWallet.connect(owner).executeTransaction(0)).to.be.revertedWith("Not enough confirmations");
    });
});
