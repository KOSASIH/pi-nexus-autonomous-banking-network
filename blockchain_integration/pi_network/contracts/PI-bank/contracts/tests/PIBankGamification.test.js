const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PIBankGamification", function () {
  let pibankGamification;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    pibankGamification = await ethers.getContractFactory("PIBankGamification");
    pibankGamification = await pibankGamification.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await pibankGamification.owner()).to.equal(owner.address);
    });
  });

  describe("Challenge Creation", function () {
    it("Should create a new challenge", async function () {
      await pibankGamification.createChallenge(owner.address, "Challenge 1", 100);
      const challengeId = await pibankGamification.getChallengeId(0);
      expect(challengeId).to.equal(1);
    });

    it("Should fail if a challenge already exists with the given ID", async function () {
      await pibankGamification.createChallenge(owner.address, "Challenge 1", 100);
      await expect(
        pibankGamification.createChallenge(owner.address, "Challenge 1", 100)
      ).to.be.revertedWith("Challenge already exists");
    });
  });
});
