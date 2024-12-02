const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Reputation System Contract", function () {
    let ReputationSystem, reputationSystem, user;

    beforeEach(async function () {
        [user] = await ethers.getSigners();
        ReputationSystem = await ethers.getContractFactory("ReputationSystem");
        reputationSystem = await ReputationSystem.deploy();
        await reputationSystem.deployed();
    });

    it("should allow a user to submit a review", async function () {
        await reputationSystem.connect(user).submitReview("Great service!", 5);
        const review = await reputationSystem.reviews(1);
        expect(review.comment).to.equal("Great service!");
        expect(review.rating).to.equal(5);
    });

    it("should calculate average rating correctly", async function () {
        await reputationSystem.connect(user).submitReview("Good service!", 4);
        await reputationSystem.connect(user).submitReview("Excellent service!", 5);
        const averageRating = await reputationSystem.getAverageRating();
        expect(averageRating).to.equal(4.5);
    });
});
