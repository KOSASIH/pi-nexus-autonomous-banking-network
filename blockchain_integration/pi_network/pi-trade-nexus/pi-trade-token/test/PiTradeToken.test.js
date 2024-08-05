const { expect } = require("chai");
const { ethers } = require("hardhat");
const { PiTradeToken } = require("../contracts/PiTradeToken.sol");

describe("PiTradeToken", function () {
  let piTradeToken;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    const PiTradeTokenFactory = await ethers.getContractFactory("PiTradeToken");
    piTradeToken = await PiTradeTokenFactory.deploy();
    await piTradeToken.deployed();
  });

  it("should have the correct name and symbol", async function () {
    expect(await piTradeToken.name()).to.equal("PiTrade Token");
    expect(await piTradeToken.symbol()).to.equal("PTT");
  });

  it("should have the correct total supply", async function () {
    expect(await piTradeToken.totalSupply()).to.equal(ethers.utils.parseEther("1000000"));
  });

  it("should allow the owner to mint new tokens", async function () {
    await piTradeToken.mint(user1.address, ethers.utils.parseEther("100"));
    expect(await piTradeToken.balanceOf(user1.address)).to.equal(ethers.utils.parseEther("100"));
  });

  it("should not allow non-owners to mint new tokens", async function () {
    await expect(piTradeToken.connect(user2).mint(user1.address, ethers.utils.parseEther("100"))).to.be.revertedWith(
      "Only the owner can mint new tokens"
    );
  });

  it("should allow users to transfer tokens", async function () {
    await piTradeToken.mint(user1.address, ethers.utils.parseEther("100"));
    await piTradeToken.connect(user1).transfer(user2.address, ethers.utils.parseEther("50"));
    expect(await piTradeToken.balanceOf(user1.address)).to.equal(ethers.utils.parseEther("50"));
    expect(await piTradeToken.balanceOf(user2.address)).to.equal(ethers.utils.parseEther("50"));
  });

  it("should not allow users to transfer more tokens than they have", async function () {
    await expect(piTradeToken.connect(user1).transfer(user2.address, ethers.utils.parseEther("100"))).to.be.revertedWith(
      "Insufficient balance"
    );
  });
});
