const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Supply Chain Contract", function () {
    let SupplyChain, supplyChain, owner;

    beforeEach(async function () {
        [owner] = await ethers.getSigners();
        SupplyChain = await ethers.getContractFactory("SupplyChain");
        supplyChain = await SupplyChain.deploy();
        await supplyChain.deployed();
    });

    it("should allow the owner to create a product", async function () {
        await supplyChain.connect(owner).createProduct("Product A");
        const product = await supplyChain.getProduct(1);
        expect(product[0]).to.equal("Product A");
        expect(product[1]).to.equal(owner.address);
        expect(product[2]).to.equal("Created");
    });

    it("should allow the owner to transfer a product", async function () {
        await supplyChain.connect(owner).createProduct("Product A");
        const [_, newOwner] = await ethers.getSigners();
        await supplyChain.connect(owner).transferProduct(1, newOwner.address);
        const product = await supplyChain.getProduct(1);
        expect(product[1]).to.equal(newOwner.address);
        expect(product[2]).to.equal("Transferred");
    });

    it("should not allow a non-owner to transfer a product", async function () {
        await supplyChain.connect(owner).createProduct("Product A");
        const [_, nonOwner] = await ethers.getSigners();
        await expect(supplyChain.connect(nonOwner).transferProduct(1, nonOwner.address)).to.be.revertedWith("Not the owner");
    });
});
