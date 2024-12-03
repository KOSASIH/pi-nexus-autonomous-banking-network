const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Decentralized Identity Contract", function () {
    let DecentralizedIdentity, identity, user;

    beforeEach(async function () {
        [user] = await ethers.getSigners();
        DecentralizedIdentity = await ethers.getContractFactory("DecentralizedIdentity");
        identity = await DecentralizedIdentity.deploy();
        await identity.deployed();
    });

    it("should allow a user to create an identity", async function () {
        await identity.connect(user).createIdentity("Alice", "alice@example.com", "1234567890");
        const id = await identity.getIdentity(user.address);
        expect(id[0]).to.equal("Alice");
        expect(id[1]).to.equal("alice@example.com");
        expect(id[2]).to.equal("1234567890");
    });

    it("should allow a user to update their identity", async function () {
        await identity.connect(user).createIdentity("Alice", "alice@example.com", "1234567890");
        await identity.connect(user).updateIdentity("Alice Updated", "alice_updated@example.com", "0987654321");
        const id = await identity.getIdentity(user.address);
        expect(id[0]).to.equal("Alice Updated");
        expect(id[1]).to.equal("alice_updated@example.com");
        expect(id[2]).to.equal("0987654321");
    });

    it("should not allow a user to create an identity if it already exists", async function () {
        await identity.connect(user).createIdentity("Alice", "alice@example.com", "1234567890");
        await expect(identity.connect(user).createIdentity("Alice", "alice@example.com", "1234567890")).to.be.revertedWith("Identity already exists");
    });
});
