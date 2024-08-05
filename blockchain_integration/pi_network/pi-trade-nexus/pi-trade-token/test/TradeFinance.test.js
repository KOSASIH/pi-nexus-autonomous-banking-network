const { expect } = require("chai");
const { ethers } = require("hardhat");
const { TradeFinance } = require("../contracts/TradeFinance.sol");
const { PiTradeToken } = require("../contracts/PiTradeToken.sol");

describe("TradeFinance", function () {
  let tradeFinance;
  let piTradeToken;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    const PiTradeTokenFactory = await ethers.getContractFactory("PiTradeToken");
    piTradeToken = await PiTradeTokenFactory.deploy();
    await piTradeToken.deployed();
    const TradeFinanceFactory = await ethers.getContractFactory("TradeFinance");
    tradeFinance = await TradeFinanceFactory.deploy(piTradeToken.address);
    await tradeFinance.deployed();
  });

  it("should have the correct PiTradeToken address", async function () {
    expect(await tradeFinance.piTradeToken()).to.equal(piTradeToken.address);
  });

  it("should allow users to create new trade finance contracts", async function () {
    await tradeFinance.createContract(user1.address, "Trade Finance Contract 1");
    expect(await tradeFinance.getContractCount()).to.equal(1);
  });

  it("should not allow users to create new trade finance contracts with invalid data", async function () {
    await expect(tradeFinance.createContract(user1.address, "")).to.be.revertedWith("Invalid contract data");
  });

  it("should allow users to update their trade finance data", async function () {
    await tradeFinance.createContract(user1.address, "Trade Finance Contract 1");
    await tradeFinance.updateData(user1.address, { totalValue: 100, totalQuantity: 10, averagePrice: 10 });
    expect(await tradeFinance.getData(user1.address)).to.deep.equal({ totalValue: 100, totalQuantity: 10, averagePrice: 10 });
  });

    it("should not allow users to update others' trade finance data", async function () {
    await tradeFinance.createContract(user1.address, "Trade Finance Contract 1");
    await expect(tradeFinance.connect(user2).updateData(user1.address, { totalValue: 100, totalQuantity: 10, averagePrice: 10 })).to.be.revertedWith(
      "Only the contract creator can update the data"
    );
  });

  it("should allow users to get their trade finance data", async function () {
    await tradeFinance.createContract(user1.address, "Trade Finance Contract 1");
    await tradeFinance.updateData(user1.address, { totalValue: 100, totalQuantity: 10, averagePrice: 10 });
    expect(await tradeFinance.getData(user1.address)).to.deep.equal({ totalValue: 100, totalQuantity: 10, averagePrice: 10 });
  });

  it("should not allow users to get others' trade finance data", async function () {
    await tradeFinance.createContract(user1.address, "Trade Finance Contract 1");
    await expect(tradeFinance.connect(user2).getData(user1.address)).to.be.revertedWith(
      "Only the contract creator can get the data"
    );
  });
});
