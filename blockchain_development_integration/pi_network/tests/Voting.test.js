const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Voting Contract", function () {
    let Voting, voting, owner;

    beforeEach ```javascript
(async function () {
        [owner] = await ethers.getSigners();
        Voting = await ethers.getContractFactory("Voting");
        voting = await Voting.deploy();
        await voting.deployed();
    });

    it("should allow the owner to add a candidate", async function () {
        await voting.connect(owner).addCandidate("Candidate A");
        const candidate = await voting.getCandidate(1);
        expect(candidate[0]).to.equal("Candidate A");
        expect(candidate[1]).to.equal(0);
    });

    it("should allow users to vote for a candidate", async function () {
        await voting.connect(owner).addCandidate("Candidate A");
        const [_, voter] = await ethers.getSigners();
        await voting.connect(voter).vote(1);
        const candidate = await voting.getCandidate(1);
        expect(candidate[1]).to.equal(1);
    });

    it("should not allow a user to vote more than once", async function () {
        await voting.connect(owner).addCandidate("Candidate A");
        const [_, voter] = await ethers.getSigners();
        await voting.connect(voter).vote(1);
        await expect(voting.connect(voter).vote(1)).to.be.revertedWith("You have already voted");
    });
});
