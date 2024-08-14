const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PiUSDTreasury', function () {
  let piUSDTreasury;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    piUSDTreasury = await ethers.getContractFactory('PiUSDTreasury');
    piUSDTreasury = await piUSDTreasury.deploy();
  });

  it('should have a treasury manager', async function () {
    expect(await piUSDTreasury.treasuryManager()).to.equal(owner.address);
  });

  it('should fund treasury correctly', async function () {
    await piUSDTreasury.fundTreasury(ethers.utils.parseEther('100'));
    expect(await piUSDTreasury.treasuryBalance()).to.equal(ethers.utils.parseEther('100'));
  });

  it('should withdraw from treasury correctly', async function () {
    await piUSDTreasury.fundTreasury(ethers.utils.parseEther('100'));
    await piUSDTreasury.withdrawFromTreasury(ethers.utils.parseEther('50'));
    expect(await piUSDTreasury.treasuryBalance()).to.equal(ethers.utils.parseEther('50'));
  });

  it('should update interest rate correctly', async function () {
    await piUSDTreasury.updateInterestRate(ethers.utils.parseEther('0.05'));
    expect(await piUSDTreasury.interestRate()).to.equal(ethers.utils.parseEther('0.05'));
  });

  it('should update reserve ratio correctly', async function () {
    await piUSDTreasury.updateReserveRatio(ethers.utils.parseEther('0.2'));
    expect(await piUSDTreasury.reserveRatio()).to.equal(ethers.utils.parseEther('0.2'));
  });
});
