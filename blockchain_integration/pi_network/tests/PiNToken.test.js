const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PiNToken", function () {
  let PiNToken, piNToken, owner, addr1, addr2, addrs;

  beforeEach(async function () {
    PiNToken = await ethers.getContractFactory("PiNToken");
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
    piNToken = await PiNToken.deploy(ethers.utils.parseEther("1000000"));
    await piNToken.deployed();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await piNToken.owner()).to.equal(owner.address);
    });

    it("Should assign the total supply of tokens to the owner", async function () {
      const ownerBalance = await piNToken.balanceOf(owner.address);
      expect(await piNToken.totalSupply()).to.equal(ownerBalance);
    });
  });

  describe("Transactions", function () {
    it("Should transfer tokens between accounts", async function () {
      await piNToken.transfer(addr1.address, ethers.utils.parseEther("100"));
      const addr1Balance = await piNToken.balanceOf(addr1.address);
      expect(addr1Balance).to.equal(ethers.utils.parseEther("100"));

      await addr1.sendTransaction({
        to: piNToken.address,
        value: ethers.utils.parseEther("1"),
      });
      const addr1NewBalance = await piNToken.balanceOf(addr1.address);
      expect(addr1NewBalance).to.equal(ethers.utils.parseEther("101"));
    });

    it("Should fail if sender doesn't have enough tokens", async function () {
      const initialOwnerBalance = await piNToken.balanceOf(owner.address);

      await expect(
        piNToken
          .connect(addr1)
          .transfer(owner.address, ethers.utils.parseEther("1")),
      ).to.be.revertedWith("Not enough tokens");

      expect(await piNToken.balanceOf(owner.address)).to.equal(
        initialOwnerBalance,
      );
    });
  });

  describe("Minting", function () {
    it("Should mint tokens to an address", async function () {
      await piNToken.mint(addr1.address, ethers.utils.parseEther("1000"));
      const addr1Balance = await piNToken.balanceOf(addr1.address);
      expect(addr1Balance).to.equal(ethers.utils.parseEther("1000"));
    });
  });
});
