const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PiUSD', function () {
  let piUSD;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    piUSD = await ethers.getContractFactory('PiUSD');
    piUSD = await piUSD.deploy();
  });

  it('should have a name', async function () {
    expect(await piUSD.name()).to.equal('PiUSD');
  });

  it('should have a symbol', async function () {
    expect(await piUSD.symbol()).to.equal('PiUSD');
  });

  it('should have 18 decimals', async function () {
    expect(await piUSD.decimals()).to.equal(18);
  });

  it('should mint tokens correctly', async function () {
    await piUSD.mint(user1.address, ethers.utils.parseEther('100'));
    expect(await piUSD.balanceOf(user1.address)).to.equal(ethers.utils.parseEther('100'));
  });

  it('should transfer tokens correctly', async function () {
    await piUSD.mint(user1.address, ethers.utils.parseEther('100'));
    await piUSD.connect(user1).transfer(user2.address, ethers.utils.parseEther('50'));
    expect(await piUSD.balanceOf(user2.address)).to.equal(ethers.utils.parseEther('50'));
  });

  it('should burn tokens correctly', async function () {
    await piUSD.mint(user1.address, ethers.utils.parseEther('100'));
    await piUSD.connect(user1).burn(ethers.utils.parseEther('50'));
    expect(await piUSD.balanceOf(user1.address)).to.equal(ethers.utils.parseEther('50'));
  });
});
