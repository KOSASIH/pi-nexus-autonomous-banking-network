// tests/test_staking.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Staking Tests", function () {
    let StakingContract, stakingContract;
    let QuantumCoin, quantumCoin;
    let owner, user;

    before(async function () {
        [owner, user] = await ethers.getSigners();

        // Deploy a mock ERC20 token for staking
        const MockERC20 = await ethers.getContractFactory("MockERC20");
        quantumCoin = await MockERC20.deploy("QuantumCoin", "QC", ethers.utils.parseEther("1000"));
        await quantumCoin.deployed();

        // Deploy StakingContract
        StakingContract = await ethers.getContractFactory("StakingContract");
        stakingContract = await StakingContract.deploy(quantumCoin.address);
        await stakingContract.deployed();

        // Transfer tokens to user for staking
        await quantumCoin.transfer(user.address, ethers.utils.parseEther("100"));
    });

    it("should allow user to stake tokens", async function () {
        await quantumCoin.connect(user).approve(stakingContract.address, ethers.utils.parseEther("50"));
        await stakingContract.connect(user).stake(ethers.utils.parseEther("50"));

        const stakedAmount = await stakingContract.stakedAmount(user.address);
        expect(stakedAmount).to.equal(ethers.utils.parseEther("50"));
    });
});
