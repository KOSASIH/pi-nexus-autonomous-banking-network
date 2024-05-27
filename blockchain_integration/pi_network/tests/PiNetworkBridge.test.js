const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PiNetworkBridge", function () {
  let piNetworkBridge, piToken, deployer, user;

  beforeEach(async function () {
    await deployments.fixture(["PiNetworkBridge"]);
    piNetworkBridge = await ethers.getContract("PiNetworkBridge");
    piToken = await ethers.getContractAt("IERC20", "PiToken address here"); // Replace with the actual PiToken contract address
    [deployer, user] = await ethers.getSigners();
  });

  describe("Deposit", function () {
    it("Should deposit Pi tokens", async function () {
      await piToken.transfer(deployer.address, "100000000000000000000"); // 100 Pi tokens
      await piNetworkBridge.deposit("100000000000000000", {
        from: deployer.address,
      });

      expect(await piToken.balanceOf(piNetworkBridge.address)).to.equal(
        "100000000000000000",
      );
    });
  });

  describe("Withdraw", function () {
    it("Should withdraw Pi tokens", async function () {
      await piToken.transfer(piNetworkBridge.address, "100000000000000000000"); // 100 Pi tokens
      await piNetworkBridge.withdraw("100000000000000000", {
        from: deployer.address,
      });

      expect(await piToken.balanceOf(piNetworkBridge.address)).to.equal("0");
      expect(await piToken.balanceOf(deployer.address)).to.equal(
        "100000000000000000",
      );
    });
  });
});
