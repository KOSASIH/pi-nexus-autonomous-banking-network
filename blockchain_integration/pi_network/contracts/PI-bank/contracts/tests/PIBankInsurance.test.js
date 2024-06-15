const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBankInsurance", function () {
  let pibankInsurance;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    pibankInsurance = await ethers.getContractFactory("PIBankInsurance");
    pibankInsurance = await pibankInsurance.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibankInsurance.owner()).to.equal(owner.address);
    });
  });

  describe("Insurance Policy Creation", function () {
    it("Should create a new insurance policy", async function () {
      await pibankInsurance.createPolicy(owner.address, 100);
      const policyId = await pibankInsurance.getPolicyId(0);
      expect(policyId).to.equal(1);
    });

    it("Should fail if a policy already exists with the given ID", async function () {
      await pibankInsurance.createPolicy(owner.address, 100);
      await expect(
        pibankInsurance.createPolicy(owner.address, 100)
      ).to.be.revertedWith("Policy already exists");
    });
  });
});
