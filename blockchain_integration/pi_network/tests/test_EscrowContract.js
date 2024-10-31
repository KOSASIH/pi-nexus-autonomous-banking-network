// tests/test_EscrowContract.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("EscrowContract", function () {
    let Escrow;
    let escrow;
    let owner;
    let seller;
    let buyer;
    let arbiter;

    beforeEach(async function () {
        Escrow = await ethers.getContractFactory("EscrowContract");
        [owner, seller, buyer, arbiter] = await ethers.getSigners();
        escrow = await Escrow.deploy();
        await escrow.deployed();
    });

    it("Should create an escrow", async function () {
        const tx = await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await tx.wait();
        
        const escrowDetails = await escrow.getEscrow(0);
        expect(escrowDetails.seller).to.equal(seller.address);
        expect(escrowDetails.arbiter).to.equal(arbiter.address);
        expect(escrowDetails.amount.toString()).to.equal(ethers.utils.parseEther("1.0").toString());
        expect(escrowDetails.isCompleted).to.be.false;
    });

    it("Should release funds to the seller", async function () {
        await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await escrow.connect(arbiter).releaseFunds(0);
        
        const escrowDetails = await escrow.getEscrow(0);
        expect(escrowDetails.isCompleted).to.be.true;
        expect(await ethers.provider.getBalance(seller.address)).to.be.above(0);
    });

    it("Should not allow non-arbiter to release funds", async function () {
        await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await expect(escrow.connect(buyer).releaseFunds(0)).to.be.revertedWith("Only arbiter can release funds");
    });

    it("Should allow arbiter to refund the buyer", async function () {
        await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await escrow.connect(arbiter).refund(0);
        
        const escrowDetails = await escrow.getEscrow(0);
        expect(escrowDetails.isCompleted).to.be.true;
        expect(await ethers.provider.getBalance(buyer.address)).to.be.above(0);
    });

    it("Should not allow non-arbiter to refund", async function () {
        await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await expect(escrow.connect(seller).refund(0)).to.be.revertedWith("Only arbiter can refund");
    });

    it("Should not allow releasing funds for an already completed escrow", async function () {
        await escrow.createEscrow(seller.address, arbiter.address, { value: ethers.utils.parseEther("1.0") });
        await escrow.connect(arbiter).releaseFunds(0);
        
        await expect(escrow.connect(arbiter).releaseFunds(0)).to.be.revertedWith("Escrow already completed");
    });
});
