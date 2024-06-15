const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBankFactory", function () {
  let pibankFactory;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    pibankFactory = await ethers.getContractFactory("PIBankFactory");
    pibankFactory = await pibankFactory.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibankFactory.owner()).to.equal(owner.address);
    });
  });

  describe("PIBank Creation", function () {
    it("Should create a new PIBank", async function () {
      await pibankFactory.createPIBank();
      const pibankAddress = await pibankFactory.getPIBankAddress(0);
      const pibank = await ethers.getContractAt("PIBank", pibankAddress);
      expect(await pibank.owner()).to.equal(owner.address);
    });

    it("Should fail if a PIBank already exists at the given address", async function () {
      await pibankFactory.createPIBank();
      await expect(
        pibankFactory.createPIBank()
      ).to.be.revertedWith("PIBank already exists");
    });
  });
});
