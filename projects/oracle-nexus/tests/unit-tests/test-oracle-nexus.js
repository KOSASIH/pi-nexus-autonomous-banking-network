const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('OracleNexus', () => {
  let oracleNexus;
  let owner;
  let oracle1;
  let oracle2;

  beforeEach(async () => {
    [owner, oracle1, oracle2] = await ethers.getSigners();
    const OracleNexus = await ethers.getContractFactory('OracleNexus');
    oracleNexus = await OracleNexus.deploy();
    await oracleNexus.deployed();
  });

  it('should deploy OracleNexus contract', async () => {
    expect(oracleNexus.address).to.not.be.undefined;
  });

  it('should allow oracles to register', async () => {
    await oracleNexus.connect(oracle1).registerOracle();
    expect(await oracleNexus.getOracleCount()).to.equal(1);
    await oracleNexus.connect(oracle2).registerOracle();
    expect(await oracleNexus.getOracleCount()).to.equal(2);
  });

  it('should allow oracles to send data', async () => {
    await oracleNexus.connect(oracle1).sendData('0x123456');
    expect(await oracleNexus.getData()).to.equal('0x123456');
  });

  it('should allow oracles to deploy models', async () => {
    const modelBytes = '0xabcdef';
    await oracleNexus.connect(oracle1).deployModel(modelBytes);
    expect(await oracleNexus.getModel()).to.equal(modelBytes);
  });
});
