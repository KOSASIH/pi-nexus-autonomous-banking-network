const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Staking Contract", function () {
    let Staking, staking, token, owner, staker;

    beforeEach(async function () {
        [owner, staker] = await ethers.getSigners();
        const Token = await ethers.getContractFactory("ERC20Mock");
        token = await Token.deploy("Staking Token", "STK", ethers.utils.parseEther("1000"));
        Staking = await ethers.getContractFactory("Staking");
        staking = await Staking.deploy(token.address, 1);
        await staking.deployed();
        await token.connect(owner).transfer(staker.address, ethers.utils.parseEther("100"));
        await token.connect(staker).approve(staking.address, ethers.utils.parseEther("100"));
    });

    it("should allow a user to stake tokens", async function () {
 ```javascript
        await staking.connect(staker).stake(ethers.utils.parseEther("50"));
        expect(await staking.stakedAmount(staker.address)).to.equal(ethers.utils.parseEther("50"));
    });

    it("should allow a user to withdraw staked tokens", async function () {
        await staking.connect(staker).stake(ethers.utils.parseEther("50"));
        await staking.connect(staker).withdraw(ethers.utils.parseEther("25"));
        expect(await staking.stakedAmount(staker.address)).to.equal(ethers.utils.parseEther("25"));
    });

    it("should allow a user to claim rewards", async function () {
        await staking.connect(staker).stake(ethers.utils.parseEther("50"));
        await ethers.provider.send("evm_mine"); // Mine a block to simulate time passing
        await staking.connect(staker).claimRewards();
        expect(await token.balanceOf(staker.address)).to.be.above(0); // Check that rewards were claimed
    });
});
