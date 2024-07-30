const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('SocialTradingContract', () => {
  it('should allow users to create trades', async () => {
    const [user] = await ethers.getSigners();
    const socialTradingContract = await ethers.getContractFactory('SocialTradingContract').then(f => f.deploy());
    await socialTradingContract.deployed();
    await socialTradingContract.createTrade(ethers.utils.getAddress('0x...'), 100);
    expect(await socialTradingContract.getTrades(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });

  it('should allow users to get trades', async () => {
    const [user] = await ethers.getSigners();
    const socialTradingContract = await ethers.getContractFactory('SocialTradingContract').then(f => f.deploy());
    await socialTradingContract.deployed();
    await socialTradingContract.createTrade(ethers.utils.getAddress('0x...'), 100);
    expect(await socialTradingContract.getTrades(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });
});
