const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('BridgeContract', () => {
  it('should allow users to deposit tokens', async () => {
    const [user] = await ethers.getSigners();
    const bridgeContract = await ethers.getContractFactory('BridgeContract').then(f => f.deploy());
    await bridgeContract.deployed();
    await bridgeContract.depositToken(ethers.utils.getAddress('0x...'), 100);
    expect(await bridgeContract.tokenBalances(user.address, ethers.utils.getAddress('0x...'))).to.be.equal(100);
  });

  it('should allow users to withdraw tokens', async () => {
    const [user] = await ethers.getSigners();
    const bridgeContract = await ethers.getContractFactory('BridgeContract').then(f => f.deploy());
    await bridgeContract.deployed();
    await bridgeContract.depositToken(ethers.utils.getAddress('0x...'), 100);
    await bridgeContract.withdrawToken(ethers.utils.getAddress('0x...'), 50);
    expect(await bridgeContract.tokenBalances(user.address, ethers.utils.getAddress('0x...'))).to.be.equal(50);
  });

  it('should allow admin to map tokens', async () => {
    const [admin] = await ethers.getSigners();
    const bridgeContract = await ethers.getContractFactory('BridgeContract').then(f => f.deploy());
    await bridgeContract.deployed();
    await bridgeContract.mapToken(ethers.utils.getAddress('0x...'), ethers.utils.getAddress('0x...'));
    expect(await bridgeContract.tokenMappings(ethers.utils.getAddress('0x...'))).to.be.equal(ethers.utils.getAddress('0x...'));
  });
});
