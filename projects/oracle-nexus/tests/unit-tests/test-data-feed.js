const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('DataFeed', () => {
  let dataFeed;
  let dataSource1;
  let dataSource2;

  beforeEach(async () => {
    [dataSource1, dataSource2] = await ethers.getSigners();
    const DataFeed = await ethers.getContractFactory('DataFeed');
    dataFeed = await DataFeed.deploy();
    await dataFeed.deployed();
  });

  it('should deploy DataFeed contract', async () => {
    expect(dataFeed.address).to.not.be.undefined;
  });

  it('should allow data sources to push data', async () => {
    await dataFeed.connect(dataSource1).pushData('0x123456');
    expect(await dataFeed.getLatestData()).to.equal('0x123456');
    await dataFeed.connect(dataSource2).pushData('0x789012');
    expect(await dataFeed.getLatestData()).to.equal('0x789012');
  });

  it('should allow data sources to be added and removed', async () => {
    await dataFeed.connect(dataSource1).addDataSource();
    expect(await dataFeed.getDataSources()).to.include(dataSource1.address);
    await dataFeed.connect(dataSource1).removeDataSource();
    expect(await dataFeed.getDataSources()).to.not.include(dataSource1.address);
  });

  it('should allow historical data to be retrieved', async () => {
    await dataFeed.connect(dataSource1).pushData('0x123456');
    await dataFeed.connect(dataSource2).pushData('0x789012');
    expect(await dataFeed.getHistoricalData(1)).to.equal('0x123456');
  });
});
