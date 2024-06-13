const { expect } = require("chai");
const { ethers } = require("hardhat");
const { DAO } = require("../contracts/DAO.sol");

describe("DAO Contract", function () {
  let dao;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    dao = await DAO.deploy();
  });

  it("should deploy the DAO contract", async function () {
    expect(await dao.name()).to.equal("DAO");
  });

  it("should create a new proposal", async function () {
    const proposalId = await dao.createProposal("Test proposal");
    expect(proposalId).to.be.a("number");
  });

  it("should vote on a proposal", async function () {
    const proposalId = await dao.createProposal("Test proposal");
    await dao.vote(proposalId, 1);
    expect(await dao.getProposalVotes(proposalId)).to.equal(1);
  });

  it("should execute a proposal", async function () {
    const proposalId = await dao.createProposal("Test proposal");
    await dao.vote(proposalId, 1);
    await dao.executeProposal(proposalId);
    expect(await dao.getProposalExecuted(proposalId)).to.be.true;
  });

  it("should transfer tokens", async function () {
    await dao.transfer(user1.address, 100);
    expect(await dao.balanceOf(user1.address)).to.equal(100);
  });
});
