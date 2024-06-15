const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBankGovernance", function () {
  let pibankGovernance;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2,...addrs] = await ethers.getSigners();
    pibankGovernance = await ethers.getContractFactory("PIBankGovernance");
    pibankGovernance = await pibankGovernance.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibankGovernance.owner()).to.equal(owner.address);
    });
  });

  describe("Proposal Creation", function () {
    it("Should create a new proposal", async function () {
      await pibankGovernance.propose(owner.address, 1);
      const proposalId = await pibankGovernance.getProposalId(0);
      expect(proposalId).to.equal(1);
    });

    it("Should fail if a proposal already exists with the given ID", async function () {
      await pibankGovernance.propose(owner.address, 1);
      await expect(
        pibankGovernance.propose(owner.address, 1)
      ).to.be.revertedWith("Proposal already exists");
    });
  });
});
