const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MyToken", () => {
  let myToken;
  let owner;
  let account1;
  let account2;

  beforeEach(async () => {
    [owner, account1, account2] = await ethers.getSigners();
    myToken = await ethers.getContractFactory("MyToken");
    myToken = await myToken.deploy();
  });

  it("should have a name", async () => {
    expect(await myToken.name()).to.equal("My Token");
  });

  it("should have a symbol", async () => {
    expect(await myToken.symbol()).to.equal("MYT");
  });

  it("should have a total supply", async () => {
    expect(await myToken.totalSupply()).to.equal(
      ethers.utils.parseEther("1000000"),
    );
  });

  it("should transfer tokens correctly", async () => {
    const amount = ethers.utils.parseEther("100");
    await myToken.transfer(account1.address, amount);
    expect(await myToken.balanceOf(account1.address)).to.equal(amount);
  });

  it("should not allow unauthorized transfers", async () => {
    const amount = ethers.utils.parseEther("100");
    await expect(
      myToken.connect(account2).transfer(account1.address, amount),
    ).to.be.revertedWith("Only the owner can transfer tokens");
  });
});
