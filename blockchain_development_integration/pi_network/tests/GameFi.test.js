const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("GameFi Contract", function () {
    let GameFi, gameFi, player;

    beforeEach(async function () {
        [player] = await ethers.getSigners();
        GameFi = await ethers.getContractFactory("GameFi");
        gameFi = await GameFi.deploy();
        await gameFi.deployed();
    });

    it("should allow a player to join a game", async function () {
        await gameFi.connect(player).joinGame();
        expect(await gameFi.players(player.address)).to.be.true;
    });

    it("should allow a player to earn rewards", async function () {
        await gameFi.connect(player).joinGame();
        await gameFi.connect(player).earnRewards(100);
        expect(await gameFi.getRewards(player.address)).to.equal(100);
    });
});
