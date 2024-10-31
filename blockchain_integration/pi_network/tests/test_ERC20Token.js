// tests/test_ERC20Token.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ERC20Token", function () {
    let Token;
    let token;
    let owner;
    let addr1;
    let addr2;

    beforeEach(async function () {
        Token = await ethers.getContractFactory("ERC20Token");
        [owner, addr1, addr2] = await ethers.getSigners();
        token = await Token.deploy("Test Token", "TTK", 1000);
        await token.deployed();
    });

    it("Should have the correct name and symbol", async function () {
        expect(await token.name()).to.equal("Test Token");
        expect(await token.symbol()).to.equal("TTK");
    });

    it("Should assign the total supply to the owner", async function () {
        const ownerBalance = await token.balanceOf(owner.address);
        expect(await token.totalSupply()).to.equal(ownerBalance);
    });

    it("Should transfer tokens between accounts", async function () {
        await token.transfer(addr1.address, 100);
        const addr1Balance = await token.balanceOf(addr1.address);
        expect(addr1Balance).to.equal(100);

        await token.connect(addr1).transfer(addr2.address, 50);
        const addr2Balance = await token.balanceOf(addr2.address);
        expect(addr2Balance).to.equal(50);
    });

    it("Should fail if sender doesn’t have enough tokens", async function () {
        const initialOwnerBalance = await token.balanceOf(owner.address);
        await expect(token.connect(addr1).transfer(owner.address, 1)).to.be.revertedWith("ERC20: transfer amount exceeds balance");
        expect(await token.balanceOf(owner.address)).to.equal(initialOwnerBalance);
    });
});
