const { expect } = require("chai");
const { polkadotApi } = require("@polkadot/api");

describe("Polkadot Token", function () {
    let polkadotToken;
    let alice;
    let bob;

    beforeEach(async function () {
        polkadotToken = await polkadotApi.createToken("MyToken", "MTK", 1000);
        alice = await polkadotApi.createAccount("Alice");
        bob = await polkadotApi.createAccount("Bob");
    });

    it("should create a new token", async function () {
        expect(await polkadotToken.name()).to.equal("MyToken");
        expect(await polkadotToken.symbol()).to.equal("MTK");
        expect(await polkadotToken.totalSupply()).to.equal(1000);
    });

    it("should transfer tokens correctly", async function () {
        await polkadotToken.transfer(alice.address, 100);
        expect(await polkadotToken.balanceOf(alice.address)).to.equal(100);

        await polkadotToken.transfer(bob.address, 50);
        expect(await polkadotToken.balanceOf(bob.address)).to.equal(50);
    });

    it("should approve tokens correctly", async function () {
        await polkadotToken.approve(alice.address, 100);
        expect(await polkadotToken.allowance(alice.address, bob.address)).to.equal(100);
    });
});
