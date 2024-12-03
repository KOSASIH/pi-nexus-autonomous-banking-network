const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("DEX Contract", function () {
    let DEX, dex, tokenA, tokenB, owner, user;

    beforeEach(async function () {
        [owner, user] = await ethers.getSigners();
        const Token = await ethers.getContractFactory("ERC20Mock");
        tokenA = await Token.deploy("Token A", "TKA", ethers.utils.parseEther("1000"));
        tokenB = await Token.deploy("Token B", "TKB", ethers.utils.parseEther("1000"));
        DEX = await ethers.getContractFactory("DEX");
        dex = await DEX.deploy();
        await dex.deployed();
    });

    it("should allow users to swap tokens", async function () {
        await tokenA.connect(owner).transfer(user.address, ethers.utils.parseEther("100"));
        await tokenA.connect(user).approve(dex.address, ethers.utils.parseEther("100"));
        await dex.swapTokens(tokenA.address, tokenB.address , ethers.utils.parseEther("100"));
        expect(await tokenB.balanceOf(user.address)).to.equal(ethers.utils.parseEther("100"));
    });

    it("should not allow swapping more tokens than the user has", async function () {
        await expect(dex.swapTokens(tokenA.address, tokenB.address, ethers.utils.parseEther("200"))).to.be.revertedWith("Insufficient balance");
    });
});
