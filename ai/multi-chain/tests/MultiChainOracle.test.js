const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('MultiChainOracle', function () {
  let multiChainOracle;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    const MultiChainOracle = await ethers.getContractFactory('MultiChainOracle');
    multiChainOracle = await MultiChainOracle.deploy();
  });

  describe('deployment', function () {
    it('should set the right owner', async function () {
      expect(await multiChainOracle.owner()).to.equal(owner.address);
    });
  });

  describe('prices', function () {
    it('should allow the owner to set prices', async function () {
      await multiChainOracle.setPrice('token1', 100);
      const price = await multiChainOracle.getPrice('token1');
      expect(price).to.equal(100);
    });

    it('should fail if the sender is not the owner', async function () {
      await expect(
        multiChainOracle.connect(user1).setPrice('token1', 100)
      ).to.be.revertedWith('Ownable: caller is not the owner');
    });
  });
});
