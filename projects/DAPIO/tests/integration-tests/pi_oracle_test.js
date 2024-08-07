const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Pi Oracle", function () {
    let piOracle;
    let owner;

    beforeEach(async function () {
        [owner] = await ethers.getSigners();
        piOracle = await ethers.deploy("PiOracle");
    });

    it("should return the correct price", async function () {
        const tokenAddress = "0x0000000000000000000000000000000000000001";
        const price = await piOracle.getPrice(tokenAddress);
        expect(price).to.equal(100);
    });

    it("should update the price correctly", async function () {
        const tokenAddress = "0x0000000000000000000000000000000000000001";
        const newPrice = 200;
        await piOracle.updatePrice(tokenAddress, newPrice);
        const updatedPrice = await piOracle.getPrice(tokenAddress);
        expect(updatedPrice).to.equal(newPrice);
    });
});
