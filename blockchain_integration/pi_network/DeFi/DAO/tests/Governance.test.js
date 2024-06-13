const { expect } = require("chai");
const { ethers } = require("hardhat");
const { Governance } = require("../contracts/Governance.sol");

describe("Governance Contract", function () {
  let governance;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    governance = await Governance.deploy();
  });

  it("should deploy the Governance contract", async function () {
    expect(await governance.name()).to.equal("Governance");
  });

  it("should update a user's role", async function () {
    await governance.updateUserRole(user1.address, 1);
    expect(await governance.getUserRole(user1.address)).to.equal(1);
  });

  it("should update a role's permissions", async function () {
    await governance.updateRolePermissions(1, 2);
    expect(await governance.getRolePermissions(1)).to.equal(2);
  });

  it("should get a user's role", async function () {
    await governance.updateUserRole(user1.address, 1);
    expect(await governance.getUserRole(user1.address)).to.equal(1);
  });

  it("should get a role's permissions", async function () {
    await governance.updateRolePermissions(1, 2);
    expect(await governance.getRolePermissions(1)).to.equal(2);
  });
});
