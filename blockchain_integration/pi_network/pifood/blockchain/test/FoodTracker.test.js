const { expect } = require('chai');
const { ethers } = require('ethers');
const { deployContract } = require('@openzeppelin/truffle-deployer');
const FoodTracker = require('../contracts/FoodTracker.sol');

describe('FoodTracker', () => {
  let foodTracker;
  let owner;
  let user1;
  let user2;

  beforeEach(async () => {
    [owner, user1, user2] = await ethers.getSigners();
    foodTracker = await deployContract('FoodTracker', FoodTracker.bytecode, FoodTracker.abi, owner);
  });

  it('should allow owner to add food item', async () => {
    const foodItem = 'Apple';
    await foodTracker.addFoodItem(foodItem);
    const foodItems = await foodTracker.getFoodItems();
    expect(foodItems).to.include(foodItem);
  });

  it('should not allow non-owner to add food item', async () => {
    const foodItem = 'Banana';
    await expect(foodTracker.connect(user1).addFoodItem(foodItem)).to.be.revertedWith('Only the owner can add food items');
  });

  it('should allow user to track food item', async () => {
    const foodItem = 'Carrot';
    await foodTracker.addFoodItem(foodItem);
    await foodTracker.connect(user1).trackFoodItem(foodItem);
    const trackedFoodItems = await foodTracker.getTrackedFoodItems(user1.address);
    expect(trackedFoodItems).to.include(foodItem);
  });

  it('should not allow user to track non-existent food item', async () => {
    const foodItem = 'Dragon Fruit';
    await expect(foodTracker.connect(user1).trackFoodItem(foodItem)).to.be.revertedWith('Food item does not exist');
  });
});
