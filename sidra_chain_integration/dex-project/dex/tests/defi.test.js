const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('DeFiContract', () => {
  it('should allow users to deposit funds', async () => {
    const [user] = await ethers.getSigners();
    const deFiContract = await ethers.getContractFactory('DeFiContract').then(f => f.deploy());
    await deFiContract.deployed();
    await deFiContract.deposit(100);
    expect(await deFiContract.getBalance()).to.be.equal(100);
  });

  it('should allow users to withdraw funds', async () => {
    const [user] = await ethers.getSigners();
    const deFiContract = await ethers.getContractFactory('DeFiContract').then(f => f.deploy());
    await deFiContract.deployed();
    await deFiContract.deposit(100);
    await deFiContract.withdraw(50);
    expect(await deFiContract.getBalance()).to.be.equal(50);
  });

  it('should allow users to get their balance', async () => {
    const [user] = await ethers.getSigners();
    const deFiContract = await ethers.getContractFactory('DeFiContract').then(f => f.deploy());
    await deFiContract.deployed();
    await deFiContract.deposit(100);
    expect(await deFiContract.getBalance()).to.be.equal(100);
  });
});
