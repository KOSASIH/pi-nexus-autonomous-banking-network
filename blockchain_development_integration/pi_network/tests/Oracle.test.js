const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Oracle Contract", function () {
    let Oracle, oracle, owner;

    beforeEach(async function () {
        [owner] = await ethers.getSigners();
        Oracle = await ethers.getContractFactory("Oracle");
        oracle = await Oracle.deploy();
        await oracle.deployed();
    });

    it("should allow the owner to set a price", async function () {
        await oracle.connect(owner).setPrice(100);
        expect(await oracle.getPrice()).to.equal(100);
    });

    it("should not allow non-owner to set a price", async function () {
        const [, nonOwner] = await ethers.getSigners();
        await expect(oracle.connect(nonOwner).setPrice(100)).to.be.revertedWith("Only owner can set price");
    });
});
