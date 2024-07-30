const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('GovernanceContract', () => {
  it('should allow admin to propose', async () => {
    const [admin, voter] = await ethers.getSigners();
    const governanceContract = await ethers.getContractFactory('GovernanceContract').then(f => f.deploy());
    await governanceContract.deployed();
    await governanceContract.propose(admin.address);
    expect(await governanceContract.proposals(admin.address)).to.be.true;
  });

  it('should allow voters to vote', async () => {
    const [admin, voter] = await ethers.getSigners();
    const governanceContract = await ethers.getContractFactory('GovernanceContract').then(f => f.deploy());
    await governanceContract.deployed();
    await governanceContract.vote(admin.address);
    expect(await governanceContract.votes(admin.address)).to.be.true;
  });
});
