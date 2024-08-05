const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PiStablecoin', () => {
  let piStablecoin;
  let piOracle;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    [owner, user1, user2] = await ethers.getSigners();

    // Deploy PiOracle contract
    const PiOracle = await ethers.getContractFactory('PiOracle');
    piOracle = await PiOracle.deploy();
    await piOracle.deployed();

    // Deploy PiStablecoin contract
    const PiStablecoin = await ethers.getContractFactory('PiStablecoin');
    piStablecoin = await PiStablecoin.deploy(piOracle.address);
    await piStablecoin.deployed();
  });

  describe('Deployment', () => {
    it('should set the owner', async () => {
      expect(await piStablecoin.owner()).to.equal(owner.address);
    });

    it('should set the PiOracle contract', async () => {
      expect(await piStablecoin.piOracle()).to.equal(piOracle.address);
    });
  });

  describe('Minting', () => {
    it('should mint Pi tokens to the owner', async () => {
      const amount = ethers.utils.parseEther('100');
      await piStablecoin.mint(owner.address, amount);
      expect(await piStablecoin.balanceOf(owner.address)).to.equal(amount);
    });

    it('should not mint Pi tokens to a non-owner', async () => {
      const amount = ethers.utils.parseEther('100');
      await expect(piStablecoin.connect(user1).mint(user1.address, amount)).to.be.revertedWith('Only the owner can mint');
    });
  });

  describe('Burning', () => {
    it('should burn Pi tokens from the owner', async () => {
      const amount = ethers.utils.parseEther('100');
      await piStablecoin.mint(owner.address, amount);
      await piStablecoin.burn(owner.address, amount);
      expect(await piStablecoin.balanceOf(owner.address)).to.equal(0);
    });

    it('should not burn Pi tokens from a non-owner', async () => {
      const amount = ethers.utils.parseEther('100');
      await expect(piStablecoin.connect(user1).burn(user1.address, amount)).to.be.revertedWith('Only the owner can burn');
    });
  });

  describe('Transferring', () => {
    it('should transfer Pi tokens between users', async () => {
      const amount = ethers.utils.parseEther('100');
      await piStablecoin.mint(owner.address, amount);
      await piStablecoin.transfer(user1.address, amount);
      expect(await piStablecoin.balanceOf(user1.address)).to.equal(amount);
    });

    it('should not transfer Pi tokens if the sender does not have enough balance', async () => {
      const amount = ethers.utils.parseEther('100');
      await expect(piStablecoin.connect(user1).transfer(user2.address, amount)).to.be.revertedWith('Insufficient balance');
    });
  });

  describe('Pi price', () => {
    it('should get the current Pi price from the PiOracle contract', async () => {
      const piPrice = await piStablecoin.getPiPrice();
      expect(piPrice).to.equal(await piOracle.getPiPrice());
    });
  });
});
