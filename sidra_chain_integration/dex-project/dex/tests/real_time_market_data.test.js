const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('MarketDataContract', () => {
  it('should allow users to update market data', async () => {
    const [user] = await ethers.getSigners();
    const marketDataContract = await ethers.getContractFactory('MarketDataContract').then(f => f.deploy());
    await marketDataContract.deployed();
    await marketDataContract.updateMarketData(ethers.utils.getAddress('0x...'), 100);
    expect(await marketDataContract.getMarketData(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });

  it('should allow users to get market data', async () => {
    const [user] = await ethers.getSigners();
    const marketDataContract = await ethers.getContractFactory('MarketDataContract').then(f => f.deploy());
    await marketDataContract.deployed();
    await marketDataContract.updateMarketData(ethers.utils.getAddress('0x...'), 100);
    expect(await marketDataContract.getMarketData(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });
});
