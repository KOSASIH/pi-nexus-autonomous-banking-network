// tests/test_VotingContract.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("VotingContract", function () {
    let Voting;
    let voting;
    let owner;
    let addr1;

    beforeEach(async function () {
        Voting = await ethers.getContractFactory("VotingContract");
        [owner, addr1] = await ethers.getSigners();
        voting = await Voting.deploy(["Option1", "Option2"]);
        await voting.deployed();
    });

    it("Should allow users to vote", async function () {
        await voting.connect(addr1).vote(0);
        expect(await voting.getVotes(0)).to.equal(1);
    });

    it("Should not allow voting for an invalid option", async function () {
        await expect(voting.connect(addr1).vote(2)).to.be.revertedWith("Invalid option");
    });

    it("Should not allow double voting", async function () {
        await voting.connect(addr1).vote(0);
        await expect(voting.connect(addr1).vote(0)).to.be.revertedWith("You have already voted");
    });
});
