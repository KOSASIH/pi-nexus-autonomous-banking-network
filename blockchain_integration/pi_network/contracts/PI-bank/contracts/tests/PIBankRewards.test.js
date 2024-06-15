const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBankRewards", function () {
  let pibankRewards;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    pibankRewards = await ethers.getContractFactory("PIBankRewards");
    pibankRewards = await pibankRewards.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibankRewards.owner()).to.equal(owner.address);
    });
  });

  describe("Reward Distribution", function () {
    it("Should distribute rewards to users", async function () {
      await pibankRewards.distributeRewards([addr1.address, addr2.address], [10, 20]);
      const addr1Balance = await pibankRewards.balanceOf(addr1.address);
      expect(addr1Balance).to.equal(10);
      const addr2Balance = await pibankRewards.balanceOf(addr2.address);
      expect(addr2Balance).to.equal(20);
    });
  });
});
