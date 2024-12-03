const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Token Vesting Contract", function () {
    let TokenVesting, vesting, token, owner, beneficiary;

    beforeEach(async function () {
        [owner, beneficiary] = await ethers.getSigners();
        const Token = await ethers.getContractFactory("ERC20Mock");
        token = await Token.deploy("Vesting Token", "VTK", ethers.utils.parseEther("1000"));
        TokenVesting = await ethers.getContractFactory("TokenVesting");
        vesting = await TokenVesting.deploy(token.address, beneficiary.address, Math.floor(Date.now() / 1000) + 60);
        await vesting.deployed();
    });

    it("should allow the owner to start vesting", async function () {
        await token.connect(owner).transfer(vesting.address, ethers.utils.parseEther("100"));
        await vesting.startVesting();
        expect(await vesting.vestingStarted()).to.be.true;
    });

    it("should allow the beneficiary to claim tokens after the vesting period", async function () {
        await token.connect(owner).transfer(vesting.address, ethers.utils.parseEther("100"));
        await vesting.startVesting();
        await ethers.provider.send("evm_increaseTime", [61]); // Increase time by 61 seconds
        await ethers.provider.send("evm_mine"); // Mine a new block
        await vesting.connect(beneficiary).claimTokens();
        expect(await token.balanceOf(beneficiary.address)).to.equal(ethers.utils.parseEther("100"));
    });
});
