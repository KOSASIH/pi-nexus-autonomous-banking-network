const { expect } = require("chai");
const { ethers } = require("hardhat");
const { Token } = require("../contracts/Token.sol");

describe("Token Contract", function () {
  let token;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    token = await Token.deploy();
  });

  it("should deploy the Token contract", async function () {
    expect(await token.name()).to.equal("Token");
  });

  it("should mint tokens", async function () {
    await token.mint(user1.address, 100);
    expect(await token.balanceOf(user1.address)).to.equal(100);
  });

  it("should burn tokens", async function () {
    await token.mint(user1.address, 100);
    await token.burn(user1.address, 50);
    expect(await token.balanceOf(user1.address)).to.equal(50);
  });

  it("should transfer tokens", async function () {
    await token.mint(user1.address, 100);
    await token.transfer(user1.address, user2.address, 50);
    expect(await token.balanceOf(user2.address)).to.equal(50);
  });
});
