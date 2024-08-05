const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('IdentityContract', () => {
  it('should allow users to create identities', async () => {
    const [user] = await ethers.getSigners();
    const identityContract = await ethers.getContractFactory('IdentityContract').then(f => f.deploy());
    await identityContract.deployed();
    await identityContract.createIdentity(ethers.utils.hexlify(ethers.utils.randomBytes(32)));
    expect(await identityContract.identities(user.address)).to.not.be.equal(0);
  });

  it('should allow users to verify identities', async () => {
    const [user] = await ethers.getSigners();
    const identityContract = await ethers.getContractFactory('IdentityContract').then(f => f.deploy());
    await identityContract.deployed();
    const identity = ethers.utils.hexlify(ethers.utils.randomBytes(32));
    await identityContract.createIdentity(identity);
    expect(await identityContract.verifyIdentity(user.address, identity)).to.be.true;
  });
});
