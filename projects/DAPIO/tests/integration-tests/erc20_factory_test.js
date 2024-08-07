const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ERC20 Factory", function () {
    let erc20Factory;
    let owner;

    beforeEach(async function () {
        [owner] = await ethers.getSigners();
        erc20Factory = await ethers.deploy("ERC20Factory");
    });

    it("should create a new ERC20 token", async function () {
        const tokenName = "MyToken";
        const tokenSymbol = "MTK";
        const totalSupply = 1000;

        const tx = await erc20Factory.createToken(tokenName, tokenSymbol, totalSupply);
        const receipt = await tx.wait();

        const tokenAddress = receipt.events[0].args.token;
        const token = await ethers.getContractAt("ERC20", tokenAddress);

        expect(await token.name()).to.equal(tokenName);
        expect(await token.symbol()).to.equal(tokenSymbol);
        expect(await token.totalSupply()).to.equal(totalSupply);
    });

    it("should return the correct token address", async function () {
        const tokenName = "MyToken";
        const tokenSymbol = "MTK";
        const totalSupply = 1000;

        const tx = await erc20Factory.createToken(tokenName, tokenSymbol, totalSupply);
        const receipt = await tx.wait();

        const tokenAddress = receipt.events[0].args.token;
        expect(await erc20Factory.getToken(owner.address)).to.equal(tokenAddress);
    });
});
