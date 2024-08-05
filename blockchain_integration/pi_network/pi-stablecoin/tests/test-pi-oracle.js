const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PiOracle', () => {
  let piOracle;
  let owner;
  let oracleNode1;
  let oracleNode2;

  beforeEach(async () => {
    [owner, oracleNode1, oracleNode2] = await ethers.getSigners();

    // Deploy PiOracle contract
    const PiOracle = await ethers.getContractFactory('PiOracle');
    piOracle = await PiOracle.deploy();
    await piOracle.deployed();
  });

  describe('Deployment', () => {
    it('should set the owner', async () => {
      expect(await piOracle.owner()).to.equal(owner.address);
    });
  });

  describe('Adding oracle nodes', () => {
    it('should add an oracle node', async () => {
      await piOracle.addOracleNode(oracleNode1.address);
      expect(await piOracle.oracleNodes(oracleNode1.address)).to.be.true;
    });

    it('should not add an oracle node if the sender is not the owner', async () => {
      await expect(piOracle.connect(oracleNode1).addOracleNode(oracleNode2.address)).to.be.revertedWith('Only the owner can add oracle nodes');
    });
  });

    describe('Removing oracle nodes', () => {
    it('should remove an oracle node', async () => {
      await piOracle.addOracleNode(oracleNode1.address);
      await piOracle.removeOracleNode(oracleNode1.address);
      expect(await piOracle.oracleNodes(oracleNode1.address)).to.be.false;
    });

    it('should not remove an oracle node if the sender is not the owner', async () => {
      await piOracle.addOracleNode(oracleNode1.address);
      await expect(piOracle.connect(oracleNode1).removeOracleNode(oracleNode1.address)).to.be.revertedWith('Only the owner can remove oracle nodes');
    });
  });

  describe('Updating Pi price', () => {
    it('should update the Pi price', async () => {
      const newPiPrice = ethers.utils.parseEther('100');
      await piOracle.updatePiPrice(newPiPrice);
      expect(await piOracle.getPiPrice()).to.equal(newPiPrice);
    });

    it('should not update the Pi price if the sender is not an oracle node', async () => {
      const newPiPrice = ethers.utils.parseEther('100');
      await expect(piOracle.connect(owner).updatePiPrice(newPiPrice)).to.be.revertedWith('Only oracle nodes can update the Pi price');
    });
  });

  describe('Getting Pi price', () => {
    it('should get the current Pi price', async () => {
      const piPrice = await piOracle.getPiPrice();
      expect(piPrice).to.be.above(0);
    });
  });
});
