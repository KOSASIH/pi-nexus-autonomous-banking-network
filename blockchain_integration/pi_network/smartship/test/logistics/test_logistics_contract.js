const { expect } = require('chai');
const { ethers } = require('hardhat');
const { LogisticsContract } = require('../blockchain_integration/contracts');
const { getProvider } = require('../utils/provider');

describe('LogisticsContract', () => {
  let logisticsContract;
  let provider;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    provider = getProvider();
    [owner, user1, user2] = await ethers.getSigners();

    // Deploy the LogisticsContract
    const logisticsContractFactory = await ethers.getContractFactory('LogisticsContract');
    logisticsContract = await logisticsContractFactory.deploy();
    await logisticsContract.deployed();

    // Set the contract address in the provider
    provider.setContractAddress('LogisticsContract', logisticsContract.address);
  });

  describe('addShipmentTracking', () => {
    it('should add a new shipment tracking', async () => {
      const shipmentId = 'SHIP-1234';
      const trackingNumber = 'TRACK-1234';
      const expectedStatus = 'IN_TRANSIT';

      await logisticsContract.connect(owner).addShipmentTracking(shipmentId, trackingNumber);

      const shipment = await logisticsContract.getShipment(shipmentId);
      expect(shipment.trackingNumber).to.equal(trackingNumber);
      expect(shipment.status).to.equal(expectedStatus);
    });

    it('should revert if shipment ID is already in use', async () => {
      const shipmentId = 'SHIP-1234';
      const trackingNumber = 'TRACK-1234';

      await logisticsContract.connect(owner).addShipmentTracking(shipmentId, trackingNumber);

      await expect(
        logisticsContract.connect(owner).addShipmentTracking(shipmentId, 'TRACK-5678')
      ).to.be.revertedWith('Shipment ID already in use');
    });
  });

  describe('updateShipmentStatus', () => {
    it('should update the shipment status', async () => {
      const shipmentId = 'SHIP-1234';
      const newStatus = 'DELIVERED';

      await logisticsContract.connect(owner).addShipmentTracking(shipmentId, 'TRACK-1234');
      await logisticsContract.connect(owner).updateShipmentStatus(shipmentId, newStatus);

      const shipment = await logisticsContract.getShipment(shipmentId);
      expect(shipment.status).to.equal(newStatus);
    });

    it('should revert if shipment ID is not found', async () => {
      const shipmentId = 'SHIP-5678';
      const newStatus = 'DELIVERED';

      await expect(
        logisticsContract.connect(owner).updateShipmentStatus(shipmentId, newStatus)
      ).to.be.revertedWith('Shipment ID not found');
    });
  });

  describe('getShipment', () => {
    it('should return the shipment details', async () => {
      const shipmentId = 'SHIP-1234';
      const trackingNumber = 'TRACK-1234';
      const expectedStatus = 'IN_TRANSIT';

      await logisticsContract.connect(owner).addShipmentTracking(shipmentId, trackingNumber);

      const shipment = await logisticsContract.getShipment(shipmentId);
      expect(shipment.trackingNumber).to.equal(trackingNumber);
      expect(shipment.status).to.equal(expectedStatus);
    });

    it('should revert if shipment ID is not found', async () => {
      const shipmentId = 'SHIP-5678';

      await expect(logisticsContract.getShipment(shipmentId)).to.be.revertedWith('Shipment ID not found');
    });
  });
});
