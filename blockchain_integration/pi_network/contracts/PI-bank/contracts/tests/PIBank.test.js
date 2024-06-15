const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBank", function () {
  let pibank;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    pibank = await ethers.getContractFactory("PIBank");
    pibank = await pibank.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibank.owner()).to.equal(owner.address);
    });

    it("Should assign the total supply of tokens to the owner", async function () {
      const ownerBalance = await pibank.balanceOf(owner.address);
      expect(await pibank.totalSupply()).to.equal(ownerBalance);
    });
  });

  describe("Transactions", function () {
    it("Should transfer tokens between accounts", async function () {
      await pibank.transfer(addr1.address, 50);
      const addr1Balance = await pibank.balanceOf(addr1.address);
      expect(addr1Balance).to.equal(50);

      await addr1.sendTransaction({ to: pibank.address, value: 50 });
      await pibank.connect(addr1).transfer(addr2.address, 50);
      const addr2Balance = await pibank.balanceOf(addr2.address);
      expect(addr2Balance).to.equal(50);
    });

    it("Should fail if sender doesn't have enough tokens", async function () {
      const initialOwnerBalance = await pibank.balanceOf(owner.address);

      await expect(
        pibank.connect(addr1).transfer(owner.address, 1)
      ).to.be.revertedWith("Not enough tokens");

      expect(await pibank.balanceOf(owner.address)).to.equal(
        initialOwnerBalance
      );
    });
  });
});
