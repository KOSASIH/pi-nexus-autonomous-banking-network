const { expect } = require("chai");
const { ethers } = require("hardhat");
const { abi, bytecode } = require("../contracts/TradeFinance.json");
const { abi: tokenAbi, bytecode: tokenBytecode } = require("../contracts/PiTradeToken.json");

describe("TradeFinance contract", () => {
  let tradeFinanceContract;
  let piTradeTokenContract;
  let deployer;
  let buyer;
  let seller;

  beforeEach(async () => {
    [deployer, buyer, seller] = await ethers.getSigners();

    // Deploy PiTradeToken contract
    const piTradeTokenFactory = new ethers.ContractFactory(tokenAbi, tokenBytecode);
    piTradeTokenContract = await piTradeTokenFactory.deploy();
    await piTradeTokenContract.deployed();

    // Deploy TradeFinance contract
    const tradeFinanceFactory = new ethers.ContractFactory(abi, bytecode);
    tradeFinanceContract = await tradeFinanceFactory.deploy(piTradeTokenContract.address);
    await tradeFinanceContract.deployed();
  });

  it("should create a new trade finance agreement", async () => {
    const agreementId = 1;
    const amount = ethers.utils.parseEther("10.0");
    const expirationDate = Math.floor(Date.now() / 1000) + 3600;

    await tradeFinanceContract.createAgreement(agreementId, buyer.address, seller.address, amount, expirationDate);

    const agreement = await tradeFinanceContract.getAgreement(agreementId);
    expect(agreement.id).to.equal(agreementId);
    expect(agreement.buyer).to.equal(buyer.address);
    expect(agreement.seller).to.equal(seller.address);
    expect(agreement.amount).to.equal(amount);
    expect(agreement.expirationDate).to.equal(expirationDate);
  });

  it("should fulfill a trade finance agreement", async () => {
    const agreementId = 1;
    const amount = ethers.utils.parseEther("10.0");
    const expirationDate = Math.floor(Date.now() / 1000) + 3600;

    await tradeFinanceContract.createAgreement(agreementId, buyer.address, seller.address, amount, expirationDate);

    await tradeFinanceContract.fulfillAgreement(agreementId);

    const agreement = await tradeFinanceContract.getAgreement(agreementId);
    expect(agreement.fulfilled).to.be.true;
  });

  it("should transfer tokens from buyer to seller when fulfilling an agreement", async () => {
    const agreementId = 1;
    const amount = ethers.utils.parseEther("10.0");
    const expirationDate = Math.floor(Date.now() / 1000) + 3600;

    await tradeFinanceContract.createAgreement(agreementId, buyer.address, seller.address, amount, expirationDate);

    const buyerBalanceBefore = await piTradeTokenContract.balanceOf(buyer.address);
    const sellerBalanceBefore = await piTradeTokenContract.balanceOf(seller.address);

    await tradeFinanceContract.fulfillAgreement(agreementId);

    const buyerBalanceAfter = await piTradeTokenContract.balanceOf(buyer.address);
    const sellerBalanceAfter = await piTradeTokenContract.balanceOf(seller.address);

    expect(buyerBalanceAfter).to.be.lt(buyerBalanceBefore);
    expect(sellerBalanceAfter).to.be.gt(sellerBalanceBefore);
  });
});
