// PiAIMarketplace.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { constants } = ethers;

const PiAIMarketplace = artifacts.require("PiAIMarketplace");

describe("PiAIMarketplace", function () {
  let marketplace;
  let owner;
  let user1;
  let user2;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    marketplace = await PiAIMarketplace.deployed();
  });

  it("should have the correct name", async function () {
    expect(await marketplace.name()).to.equal("PiAI Marketplace");
  });

  it("should have the correct symbol", async function () {
    expect(await marketplace.symbol()).to.equal("PIAI");
  });

  it("should allow listing of AI models", async function () {
    const aiModel = {
      name: "My AI Model",
      description: "A simple AI model",
      price: ethers.utils.parseEther("1.0"),
    };

    await marketplace.listAIModel(aiModel.name, aiModel.description, aiModel.price, {
      from: user1.address,
    });

    const listedAIModels = await marketplace.getAIModels();
    expect(listedAIModels.length).to.equal(1);
    expect(listedAIModels[0].name).to.equal(aiModel.name);
  });

  it("should allow trading of AI models", async function () {
    const aiModel = {
      name: "My AI Model",
      description: "A simple AI model",
      price: ethers.utils.parseEther("1.0"),
    };

    await marketplace.listAIModel(aiModel.name, aiModel.description, aiModel.price, {
      from: user1.address,
    });

    await marketplace.tradeAIModel(aiModel.name, {
      from: user2.address,
      value: aiModel.price,
    });

    const ownerOfAIModel = await marketplace.getAIModelOwner(aiModel.name);
    expect(ownerOfAIModel).to.equal(user2.address);
  });

  it("should allow users to rate and review AI models", async function () {
    const aiModel = {
      name: "My AI Model",
      description: "A simple AI model",
      price: ethers.utils.parseEther("1.0"),
    };

    await marketplace.listAIModel(aiModel.name, aiModel.description, aiModel.price, {
      from: user1.address,
    });

    await marketplace.rateAndReviewAIModel(aiModel.name, 5, "Great AI model!", {
      from: user2.address,
    });

    const ratings = await marketplace.getAIModelRatings(aiModel.name);
    expect(ratings.length).to.equal(1);
    expect(ratings[0].rating).to.equal(5);
    expect(ratings[0].review).to.equal("Great AI model!");
  });
});
