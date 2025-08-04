const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MultiChainBankManager", function () {
  let multiChainBankManager;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    const MultiChainBankManager = await ethers.getContractFactory(
      "MultiChainBankManager",
    );
    multiChainBankManager = await MultiChainBankManager.deploy();
  });

  describe("deployment", function () {
    it("should set the right owner", async function () {
      expect(await multiChainBankManager.owner()).to.equal(owner.address);
    });
  });

  describe("management", function () {
    it("should allow only the owner to add a bank", async function () {
      await expect(
        multiChainBankManager.connect(user1).addBank(user1.address),
      ).to.be.revertedWith("Ownable: caller is not the owner");

      await multiChainBankManager.addBank(user1.address);
      const isBank = await multiChainBankManager.isBank(user1.address);
      expect(isBank).to.equal(true);
    });

    it("should allow only the owner to remove a bank", async function () {
      await multiChainBankManager.addBank(user1.address);
      const isBank = await multiChainBankManager.isBank(user1.address);
      expect(isBank).to.equal(true);

      await expect(
        multiChainBankManager.connect(user1).removeBank(user1.address),
      ).to.be.revertedWith("Ownable: caller is not the owner");

      await multiChainBankManager.removeBank(user1.address);
      const isBank = await multiChainBankManager.isBank(user1.address);
      expect(isBank).to.equal(false);
    });
  });
});
