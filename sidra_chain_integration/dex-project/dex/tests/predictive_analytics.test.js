const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PredictiveAnalyticsContract', () => {
  it('should allow users to make predictions', async () => {
    const [user] = await ethers.getSigners();
    const predictiveAnalyticsContract = await ethers.getContractFactory('PredictiveAnalyticsContract').then(f => f.deploy());
    await predictiveAnalyticsContract.deployed();
    await predictiveAnalyticsContract.makePrediction(ethers.utils.getAddress('0x...'), 100);
    expect(await predictiveAnalyticsContract.getPrediction(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });

  it('should allow users to get predictions', async () => {
    const [user] = await ethers.getSigners();
    const predictiveAnalyticsContract = await ethers.getContractFactory('PredictiveAnalyticsContract').then(f => f.deploy());
    await predictiveAnalyticsContract.deployed();
    await predictiveAnalyticsContract.makePrediction(ethers.utils.getAddress('0x...'), 100);
    expect(await predictiveAnalyticsContract.getPrediction(ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });
});
