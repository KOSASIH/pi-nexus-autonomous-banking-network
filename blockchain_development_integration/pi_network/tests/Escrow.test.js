const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Escrow Contract", function () {
    let Escrow, escrow, buyer, seller, arbiter;

    beforeEach(async function () {
        [buyer, seller, arbiter] = await ethers.getSigners();
        Escrow = await ethers.getContractFactory("Escrow");
        escrow = await Escrow.deploy(seller.address, arbiter.address);
        await escrow.deployed();
    });

    it("should allow the buyer to deposit funds", async function () {
        await escrow.connect(buyer).deposit({ value: ethers.utils.parseEther("1") });
        expect(await escrow.buyerBalance()).to.equal(ethers.utils.parseEther("1"));
    });

    it("should allow the arbiter to release funds", async function () {
        await escrow.connect(buyer).deposit({ value: ethers.utils.parseEther("1") });
        await escrow.connect(arbiter).releaseFunds();
        expect(await ethers.provider.getBalance(seller.address)).to.equal(ethers.utils.parseEther("1"));
    });

    it("should not allow non-arbiter to release funds", async function () {
        await escrow.connect(buyer).deposit({ value: ethers.utils.parseEther("1") });
        await expect(escrow.connect(buyer).releaseFunds()).to.be.revertedWith("Only arbiter can release funds");
    });
});
