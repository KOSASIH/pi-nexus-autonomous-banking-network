const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Time Locked Wallet Contract", function () {
    let TimeLockedWallet, wallet, owner, beneficiary;

    beforeEach(async function () {
        [owner, beneficiary] = await ethers.getSigners();
        TimeLockedWallet = await ethers.getContractFactory("TimeLockedWallet");
        wallet = await TimeLockedWallet.deploy(beneficiary.address, Math.floor(Date.now() / 1000) + 60);
        await wallet.deployed();
    });

    it("should allow the owner to deposit funds", async function () {
        await wallet.connect(owner).deposit({ value: ethers.utils.parseEther("1") });
        expect(await ethers.provider.getBalance(wallet.address)).to.equal(ethers.utils.parseEther("1"));
    });

    it("should not allow the beneficiary to withdraw funds before the unlock time", async function () {
        await wallet.connect(owner).deposit({ value: ethers.utils.parseEther("1") });
        await expect(wallet.connect(beneficiary).withdraw()).to.be.revertedWith("Funds are locked");
    });

    it("should allow the beneficiary to withdraw funds after the unlock time", async function () {
        await wallet.connect(owner).deposit({ value: ethers.utils.parseEther("1") });
        await ethers.provider.send("evm_increaseTime", [61]); // Increase time by 61 seconds
        await ethers.provider.send("evm_mine"); // Mine a new block
        await wallet.connect(beneficiary).withdraw();
        expect(await ethers.provider.getBalance(wallet.address)).to.equal(0);
    });
});
