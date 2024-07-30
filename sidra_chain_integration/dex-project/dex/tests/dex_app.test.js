const { expect } = require('chai');
const { ethers } = require('hardhat');
const { WalletContract } = require('../dex_app/WalletContract.sol');
const { DexAppContract } = require('../dex_app/DexAppContract.sol');

describe('DexAppContract', () => {
  let dexAppContract;
  let walletContract;
  let owner;
  let user1;
  let user2;
  let tokenContract;

  beforeEach(async () => {
    [owner, user1, user2] = await ethers.getSigners();
    tokenContract = await ethers.getContractFactory('ERC20Token').then(f => f.deploy());
    await tokenContract.deployed();
    walletContract = await ethers.getContractFactory('WalletContract').then(f => f.deploy());
    await walletContract.deployed();
    dexAppContract = await ethers.getContractFactory('DexAppContract').then(f => f.deploy());
    await dexAppContract.deployed();
  });

  it('should allow a user to place an order', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await dexAppContract.connect(user1).placeOrder(tokenContract.address, 20, 10);
    expect(await dexAppContract.orders(user1.address, tokenContract.address)).to.be.equal(20);
  });

  it('should not allow a user to place an order with insufficient balance', async () => {
    await expect(dexAppContract.connect(user1).placeOrder(tokenContract.address, 100, 10)).to.be.revertedWith('Insufficient balance');
  });

  it('should allow a user to cancel an order', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await dexAppContract.connect(user1).placeOrder(tokenContract.address, 20, 10);
    await dexAppContract.connect(user1).cancelOrder(tokenContract.address);
    expect(await dexAppContract.orders(user1.address, tokenContract.address)).to.be.equal(0);
  });

  it('should allow a user to execute a trade', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await dexAppContract.connect(user1).placeOrder(tokenContract.address, 20, 10);
    await dexAppContract.connect(user2).executeTrade(user1.address, tokenContract.address, 10, 10);
    expect(await walletContract.balanceOf(user2.address, tokenContract.address)).to.be.equal(10);
  });

  it('should not allow a user to execute a trade with insufficient liquidity', async () => {
    await expect(dexAppContract.connect(user2).executeTrade(user1.address, tokenContract.address, 100, 10)).to.be.revertedWith('Insufficient liquidity');
  });

  it('should allow a user to add liquidity to a pool', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await dexAppContract.connect(user1).addLiquidity(tokenContract.address, 20);
    expect(await dexAppContract.liquidityPools(tokenContract.address, user1.address)).to.be.equal(20);
  });

  it('should allow a user to remove liquidity from a pool', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await dexAppContract.connect(user1).addLiquidity(tokenContract.address, 20);
    await dexAppContract.connect(user1).removeLiquidity(tokenContract.address, 10);
    expect(await dexAppContract.liquidityPools(tokenContract.address, user1.address)).to.be.equal(10);
  });
});
