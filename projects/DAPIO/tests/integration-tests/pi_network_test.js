const { expect } = require("chai");
const { piNetwork } = require("@pi-network/api");

describe("Pi Network", function () {
    let piNetwork;
    let alice;
    let bob;

    beforeEach(async function () {
        piNetwork = await piNetworkApi.createNetwork("MyNetwork");
        alice = await piNetwork.createAccount("Alice");
        bob = await piNetwork.createAccount("Bob");
    });

    it("should create a new network", async function () {
        expect(await piNetwork.name()).to.equal("MyNetwork");
    });

    it("should create a new account", async function () {
        expect(await alice.address()).to.not.be.null;
        expect(await bob.address()).to.not.be.null;
    });

    it("should transfer tokens correctly", async function () {
        await piNetwork.transfer(alice.address, 100);
        expect(await piNetwork.balanceOf(alice.address)).to.equal(100);

        await piNetwork.transfer(bob.address, 50);
        expect(await piNetwork.balanceOf(bob.address)).to.equal(50);
    });
});
