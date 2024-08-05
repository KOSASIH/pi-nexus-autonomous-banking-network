const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('LiquidityPoolContract', () => {
  it('should allow users to provide liquidity', async () => {
    const [provider] = await ethers.getSigners();
    const liquidityPoolContract = await ethers.getContractFactory('LiquidityPoolContract').then(f => f.deploy());
    await liquidityPoolContract.deployed();
    await liquidityPoolContract.provideLiquidity(100);
    expect(await liquidityPoolContract.liquidityProviders(provider.address)).to.be.equal(100);
  });

  it('should allow users to withdraw liquidity', async () => {
    const [provider] = await ethers.getSigners();
    const liquidityPoolContract = await ethers.getContractFactory('LiquidityPoolContract').then(f => f.deploy());
    await liquidityPoolContract.deployed();
    await liquidityPoolContract.provideLiquidity(100);
    await liquidityPoolContract.withdrawLiquidity(50);
    expect(await liquidityPoolContract.liquidityProviders(provider.address)).to.be.equal(50);
  });
});
