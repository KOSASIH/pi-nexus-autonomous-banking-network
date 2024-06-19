const { expect } = require('chai');
const { ethers } = require('hardhat');
const { ShipmentContract } = require('../blockchain_integration/contracts');
const { getProvider } = require('../utils/provider');

describe('ShipmentContract', () => {
  let shipmentContract;
  let provider;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    provider = getProvider();
    [owner, user1, user2] = await ethers.getSigners();

    // Deploy the ShipmentContract
    const shipmentContractFactory = await ethers.getContractFactory('ShipmentContract');
    shipmentContract = await shipmentContractFactory.deploy();
    await shipmentContract.deployed();

    // Set the contract address in the provider
    provider.setContractAddress('ShipmentContract', shipmentContract.address);
  });

  describe('createShipment', () => {
    it('should create a new shipment', async () => {
      const shipmentData = {
        sender: user1.address,
        recipient: user2.address,
        shipmentType: 'PACKAGE',
        weight: 1,
        dimensions: [10, 10, 10],
      };

      await shipmentContract.connect(owner).createShipment(shipmentData);

      const shipmentId = await shipmentContract.getShipmentId(shipmentData.sender, shipmentData.recipient);
      expect(shipmentId).to.be.a('string');

      const shipment = await shipmentContract.getShipment(shipmentId);
      expect(shipment.sender).to.equal(shipmentData.sender);
      expect(shipment.recipient).to.equal(shipmentData.recipient);
      expect(shipment.shipmentType).to.equal(shipmentData.shipmentType);
      expect(shipment.weight).to.equal(shipmentData.weight);
      expect(shipment.dimensions).to.deep.equal(shipmentData.dimensions);
    });

    it('should revert if shipment data is invalid', async () => {
      const shipmentData = {
        sender: user1.address,
        recipient: user2.address,
        shipmentType: 'INVALID',
        weight: 0,
        dimensions: [10, 10, 10],
      };

      await expect(
        shipmentContract.connect(owner).createShipment(shipmentData)
      ).to.be.revertedWith('Invalid shipment data');
    });
  });

  describe('updateShipmentLocation', () => {
    it('should update the shipment location', async () => {
      const shipmentData = {
        sender: user1.address,
        recipient: user2.address,
        shipmentType: 'PACKAGE',
        weight: 1,
        dimensions: [10, 10, 10],
      };

      await shipmentContract.connect(owner).createShipment(shipmentData);

      const shipmentId = await shipmentContract.getShipmentId(shipmentData.sender, shipmentData.recipient);
      const newLocation = 'NEW_LOCATION';

      await shipmentContract.connect(owner).updateShipmentLocation(shipmentId, newLocation);

      const shipment = await shipmentContract.getShipment(shipmentId);
      expect(shipment.location).to.equal(newLocation);
    });

    it('should revert if shipment ID is not found', async () => {
      const shipmentId = 'INVALID_SHIPMENT_ID';
      const newLocation = 'NEW_LOCATION';

      await expect(
        shipmentContract.connect(owner).updateShipmentLocation(shipmentId, newLocation)
      ).to.be.revertedWith('Shipment ID not found');
    });
  });

  describe('getShipment', () => {
    it('should return the shipment details', async () => {
      const shipmentData = {
        sender: user1.address,
        recipient: user2.address,
        shipmentType: 'PACKAGE',
        weight: 1,
        dimensions: [10, 10, 10],
      };

      await shipmentContract.connect(owner).createShipment(shipmentData);

      const shipmentId = await shipmentContract.getShipmentId(shipmentData.sender, shipmentData.recipient);
      const shipment = await shipmentContract.getShipment(shipmentId);

      expect(shipment.sender).to.equal(shipmentData.sender);
      expect(shipment.recipient).to.equal(shipmentData.recipient);
      expect(shipment.shipmentType).to.equal(shipmentData.shipmentType);
      expect(shipment.weight).to.equal(shipmentData.weight);
      expect(shipment.dimensions).to.deep.equal(shipmentData.dimensions);
    });

    it('should revert if shipment ID is not found', async () => {
      const shipmentId = 'INVALID_SHIPMENT_ID';

      await expect(shipmentContract.getShipment(shipmentId)).to.be.revertedWith('Shipment ID not found');
    });
  });
});
