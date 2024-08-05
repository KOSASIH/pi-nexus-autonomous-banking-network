const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('SecurityContract', () => {
  let securityContract;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    [owner, user1, user2] = await ethers.getSigners();
    securityContract = await ethers.getContractFactory('SecurityContract').then(f => f.deploy());
    await securityContract.deployed();
  });

  it('should allow the owner to update a user\'s access level', async () => {
    await securityContract.updateAccessLevel(user1.address, 2);
    expect(await securityContract.getAccessLevel(user1.address)).to.be.equal(2);
  });

  it('should not allow a non-owner to update a user\'s access level', async () => {
    await expect(securityContract.connect(user2).updateAccessLevel(user1.address, 2)).to.be.revertedWith('Only the owner can call this function');
  });

  it('should allow a user to authenticate themselves', async () => {
    await securityContract.authenticateUser(user1.address);
    expect(await securityContract.isAuthenticated(user1.address)).to.be.true;
  });

  it('should allow the owner to revoke a user\'s authentication', async () => {
    await securityContract.authenticateUser(user1.address);
    await securityContract.revokeUserAuthentication(user1.address);
    expect(await securityContract.isAuthenticated(user1.address)).to.be.false;
  });

  it('should not allow a non-owner to revoke a user\'s authentication', async () => {
    await securityContract.authenticateUser(user1.address);
    await expect(securityContract.connect(user2).revokeUserAuthentication(user1.address)).to.be.revertedWith('Only the owner can call this function');
  });

  it('should return the correct access level for a user', async () => {
    await securityContract.updateAccessLevel(user1.address, 3);
    expect(await securityContract.getAccessLevel(user1.address)).to.be.equal(3);
  });

  it('should return false for an unauthenticated user', async () => {
    expect(await securityContract.isAuthenticated(user2.address)).to.be.false;
  });
});
