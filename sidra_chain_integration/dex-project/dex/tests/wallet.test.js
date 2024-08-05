const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('WalletContract', () => {
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
  });

  it('should allow a user to deposit tokens into their wallet', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    expect(await walletContract.balanceOf(user1.address, tokenContract.address)).to.be.equal(50);
  });

  it('should not allow a user to withdraw more tokens than they have in their wallet', async () => {
    await expect(walletContract.connect(user1).withdraw(tokenContract.address, 100)).to.be.revertedWith('Insufficient balance');
  });

  it('should allow a user to transfer tokens to another user', async () => {
    await tokenContract.transfer(user1.address, 100);
    await walletContract.connect(user1).deposit(tokenContract.address, 50);
    await walletContract.connect(user1).transfer(user2.address, tokenContract.address, 20);
    expect(await walletContract.balanceOf(user1.address, tokenContract.address)).to.be.equal(30);
    expect(await walletContract.balanceOf(user2.address, tokenContract.address)).to.be.equal(20);
  });

  it('should allow a user to approve another user to spend tokens on their behalf', async () => {
    await walletContract.connect(user1).approve(user2.address, tokenContract.address, 50);
    expect(await walletContract.allowance(user1.address, user2.address, tokenContract.address)).to.be.equal(50);
  });

  it('should not allow a user to transfer tokens to another user if they do not have sufficient balance', async () => {
    await expect(walletContract.connect(user1).transfer(user2.address, tokenContract.address, 100)).to.be.revertedWith('Insufficient balance');
  });
});
