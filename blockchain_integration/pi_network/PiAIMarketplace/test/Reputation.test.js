// Reputation.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { constants } = ethers;

const Reputation = artifacts.require("Reputation");

describe("Reputation", function () {
  let reputation;
  let user1;
  let user2;

  beforeEach(async function () {
    [user1, user2] = await ethers.getSigners();
    reputation = await Reputation.deployed();
  });

  it("should allow users to add reviews", async function () {
    await reputation.addReview(user2.address, 5, "Great user!", {
      from: user1.address,
    });

    const reviews = await reputation.getReviews(user2.address);
    expect(reviews.length).to.equal(1);
    expect(reviews[0].rating).to.equal(5);
    expect(reviews[0].review).to.equal("Great user!");
  });

  it("should allow users to get their average rating", async function () {
    await reputation.addReview(user2.address, 5, "Great user!", {
      from: user1.address,
    });

    await reputation.addReview(user2.address, 4, "Good user!", {
      from: user1.address,
    });

    const averageRating = await reputation.getAverageRating(user2.address);
    expect(averageRating).to.equal(4.5);
  });

  it("should allow users to get their reputation score", async function () {
    await reputation.addReview(user2.address, 5, "Great user!", {
      from: user1.address,
    });

    await reputation.addReview(user2.address, 4, "Good user!", {
      from: user1.address,
    });

    const reputationScore = await reputation.getReputationScore(user2.address);
    expect(reputationScore).to.be.above(0);
  });
});
