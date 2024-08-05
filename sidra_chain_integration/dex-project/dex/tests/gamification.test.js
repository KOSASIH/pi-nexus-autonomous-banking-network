const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('GamificationContract', () => {
  it('should allow users to update scores', async () => {
    const [user] = await ethers.getSigners();
    const gamificationContract = await ethers.getContractFactory('GamificationContract').then(f => f.deploy());
    await gamificationContract.deployed();
    await gamificationContract.updateScore(100);
    expect(await gamificationContract.getScore()).to.be.equal(100);
  });

  it('should allow users to get scores', async () => {
    const [user] = await ethers.getSigners();
    const gamificationContract = await ethers.getContractFactory('GamificationContract').then(f => f.deploy());
    await gamificationContract.deployed();
    await gamificationContract.updateScore(100);
    expect(await gamificationContract.getScore()).to.be.equal(100);
  });

  it('should allow users to level up', async () => {
    const [user] = await ethers.getSigners();
    const gamificationContract = await ethers.getContractFactory('GamificationContract').then(f => f.deploy());
    await gamificationContract.deployed();
    await gamificationContract.updateScore(100);
    await gamificationContract.levelUp();
    expect(await gamificationContract.getLevel()).to.be.equal(2);
  });

  it('should allow users to get their level', async () => {
    const [user] = await ethers.getSigners();
    const gamificationContract = await ethers.getContractFactory('GamificationContract').then(f => f.deploy());
    await gamificationContract.deployed();
    await gamificationContract.updateScore(100);
    await gamificationContract.levelUp();
    expect(await gamificationContract.getLevel()).to.be.equal(2);
  });

  it('should allow users to get their rank', async () => {
    const [user] = await ethers.getSigners();
    const gamificationContract = await ethers.getContractFactory('GamificationContract').then(f => f.deploy());
    await gamificationContract.deployed();
    await gamificationContract.updateScore(100);
    await gamificationContract.levelUp();
    expect(await gamificationContract.getRank()).to.be.equal(1);
  });
});
