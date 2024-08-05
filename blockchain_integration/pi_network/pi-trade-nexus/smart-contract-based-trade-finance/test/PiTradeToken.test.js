const { expect } = require("chai");
const { ethers } = require("hardhat");
const { abi, bytecode } = require("../contracts/PiTradeToken.json");

describe("PiTradeToken contract", () => {
  let piTradeTokenContract;
  let deployer;
  let user1;
  let user2;

  beforeEach(async () => {
    [deployer, user1, user2] = await ethers.getSigners();

    // Deploy PiTradeToken contract
    const piTradeTokenFactory = new ethers.ContractFactory(abi, bytecode);
    piTradeTokenContract = await piTradeTokenFactory.deploy();
    await piTradeTokenContract.deployed();
  });

  it("should have a total supply of 1 million tokens", async () => {
    const totalSupply = await piTradeTokenContract.totalSupply();
    expect(totalSupply).to.equal(ethers.utils.parseEther("1000000.0"));
  });

  it("should transfer tokens from one user to another", async () => {
    const amount = ethers.utils.parseEther("10.0");

        await piTradeTokenContract.transfer(user2.address, amount, { from: user1.address });

    const user1BalanceAfter = await piTradeTokenContract.balanceOf(user1.address);
    const user2BalanceAfter = await piTradeTokenContract.balanceOf(user2.address);

    expect(user1BalanceAfter).to.be.lt(user1Balance);
    expect(user2BalanceAfter).to.be.gt(0);
  });

  it("should approve token spending for a user", async () => {
    const amount = ethers.utils.parseEther("10.0");

    await piTradeTokenContract.approve(user1.address, amount);

    const allowance = await piTradeTokenContract.allowance(deployer.address, user1.address);
    expect(allowance).to.equal(amount);
  });

  it("should increase the token balance of a user when minting new tokens", async () => {
    const amount = ethers.utils.parseEther("10.0");

    await piTradeTokenContract.mint(user1.address, amount);

    const user1Balance = await piTradeTokenContract.balanceOf(user1.address);
    expect(user1Balance).to.equal(amount);
  });

  it("should decrease the token balance of a user when burning tokens", async () => {
    const amount = ethers.utils.parseEther("10.0");

    await piTradeTokenContract.mint(user1.address, amount);

    const user1BalanceBefore = await piTradeTokenContract.balanceOf(user1.address);

    await piTradeTokenContract.burn(amount, { from: user1.address });

    const user1BalanceAfter = await piTradeTokenContract.balanceOf(user1.address);
    expect(user1BalanceAfter).to.be.lt(user1BalanceBefore);
  });
});
