const { expect } = require('chai');
const { ethers } = require('ethers');
const { deployContract } = require('@openzeppelin/truffle-deployer');
const InventoryManager = require('../contracts/InventoryManager.sol');

describe('InventoryManager', () => {
  let inventoryManager;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    [owner, user1, user2] = await ethers.getSigners();
    inventoryManager = await deployContract('InventoryManager', InventoryManager.bytecode, InventoryManager.abi, owner);
  });

  it('should allow owner to create inventory item', async () => {
    const itemName = 'Apple';
    const itemQuantity = 10;
    await inventoryManager.createInventoryItem(itemName, itemQuantity);
    const inventoryItems = await inventoryManager.getInventoryItems();
    expect(inventoryItems).to.include(itemName);
  });

  it('should not allow non-owner to create inventory item', async () => {
    const itemName = 'Banana';
    const itemQuantity = 10;
    await expect(inventoryManager.connect(user1).createInventoryItem(itemName, itemQuantity)).to.be.revertedWith('Only the owner can create inventory items');
  });

  it('should allow user to increment inventory item quantity', async () => {
    const itemName = 'Carrot';
    const itemQuantity = 10;
    await inventoryManager.createInventoryItem(itemName, itemQuantity);
    await inventoryManager.connect(user1).incrementInventoryItemQuantity(itemName, 5);
    const updatedQuantity = await inventoryManager.getInventoryItemQuantity(itemName);
    expect(updatedQuantity).to.equal(15);
  });

  it('should not allow user to decrement inventory item quantity below 0', async () => {
    const itemName = 'Dragon Fruit';
    const itemQuantity = 10;
    await inventoryManager.createInventoryItem(itemName, itemQuantity);
    await expect(inventoryManager.connect(user1).decrementInventoryItemQuantity(itemName, 15)).to.be.revertedWith('Cannot decrement quantity below 0');
  });
});
