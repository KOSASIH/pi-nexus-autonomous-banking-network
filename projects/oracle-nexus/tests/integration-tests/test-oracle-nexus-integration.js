const { expect } = require('chai');
const { ethers } = require('hardhat');
const { deployContract } = require('../utils/deploy');

describe('OracleNexus Integration', () => {
  let oracleNexus;
  let dataFeed;
  let oracle1;
  let oracle2;

  beforeEach(async () => {
    [oracle1, oracle2] = await ethers.getSigners();
    oracleNexus = await deployContract('OracleNexus');
    dataFeed = await deployContract('DataFeed');
    await oracleNexus.setDataFeed(dataFeed.address);
  });

  it('should integrate OracleNexus with DataFeed', async () => {
    // Register oracles with OracleNexus
    await oracleNexus.connect(oracle1).registerOracle();
    await oracleNexus.connect(oracle2).registerOracle();

    // Send data from oracles to OracleNexus
    await oracleNexus.connect(oracle1).sendData('0x123456');
    await oracleNexus.connect(oracle2).sendData('0x789012');

    // Verify data is aggregated and sent to DataFeed
    expect(await dataFeed.getLatestData()).to.equal('0x123456789012');

    // Deploy a model from OracleNexus to DataFeed
    const modelBytes = '0xabcdef';
    await oracleNexus.connect(oracle1).deployModel(modelBytes);
    expect(await dataFeed.getModel()).to.equal(modelBytes);
  });

  it('should handle multiple oracles and data sources', async () => {
    // Register multiple oracles with OracleNexus
    await oracleNexus.connect(oracle1).registerOracle();
    await oracleNexus.connect(oracle2).registerOracle();
    await oracleNexus.connect(oracle1).registerOracle(); // duplicate registration

    // Send data from multiple oracles to OracleNexus
    await oracleNexus.connect(oracle1).sendData('0x123456');
    await oracleNexus.connect(oracle2).sendData('0x789012');
    await oracleNexus.connect(oracle1).sendData('0x345678');

    // Verify data is aggregated and sent to DataFeed
    expect(await dataFeed.getLatestData()).to.equal('0x123456789012345678');

    // Add multiple data sources to DataFeed
    await dataFeed.connect(oracle1).addDataSource();
    await dataFeed.connect(oracle2).addDataSource();
    await dataFeed.connect(oracle1).addDataSource(); // duplicate addition

    // Verify data sources are correctly added and removed
    expect(await dataFeed.getDataSources()).to.include(oracle1.address);
    expect(await dataFeed.getDataSources()).to.include(oracle2.address);
    await dataFeed.connect(oracle1).removeDataSource();
    expect(await dataFeed.getDataSources()).to.not.include(oracle1.address);
  });
});
