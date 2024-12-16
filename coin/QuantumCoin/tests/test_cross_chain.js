// test_cross_chain.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("CrossChainBridge", function () {
    let CrossChainBridge;
    let bridge;
    let token;
    let owner;
    let addr1;

    beforeEach(async function () {
        // Deploy a mock ERC20 token
        const Token = await ethers.getContractFactory("MockERC20");
        token = await Token.deploy("Mock Token", "MTK", ethers.utils.parseUnits("1000", 18));
        await token.deployed();

        CrossChainBridge = await ethers.getContractFactory("CrossChainBridge");
        [owner, addr1] = await ethers.getSigners();
        bridge = await CrossChainBridge.deploy();
        await bridge.deployed();

        // Transfer some tokens to addr1 for testing
        await token.transfer(addr1.address, ethers.utils.parseUnits("100", 18));
    });

    it("Should lock assets on the bridge", async function () {
        const amount = ethers.utils.parseUnits("10", 18);
        await token.connect(addr1).approve(bridge.address, amount);
        
        await expect(bridge.lockAssets(token.address, amount, "Ethereum"))
            .to.emit(bridge, "AssetsLocked")
            .withArgs(addr1.address, amount, "Ethereum");

        expect(await bridge.lockedAssets(token.address)).to.equal(amount);
    });

    it("Should mint equivalent assets on the destination chain", async function () {
        const amount = ethers.utils.parseUnits("10", 18);
        await token.connect(addr1).approve(bridge.address, amount);
        await bridge.lockAssets(token.address, amount, "Ethereum");

        await expect(bridge.mintAssets(addr1.address, amount, "Ethereum"))
            .to.emit(bridge, "AssetsMinted")
            .withArgs(addr1.address, amount, "Ethereum");
    });

    it("Should allow the owner to withdraw locked assets", async function () {
        const amount = ethers.utils.parseUnits("10", 18);
        await token.connect(addr1).approve(bridge.address, amount);
        await bridge.lockAssets(token.address, amount, "Ethereum");

        await bridge.withdrawLockedAssets(token.address, amount);
        expect(await token.balanceOf(owner.address)).to.equal(amount);
        expect(await bridge.lockedAssets(token.address)).to.equal(0);
    });

    it("Should not allow non-owners to withdraw locked assets", async function () {
        const amount = ethers.utils.parseUnits("10",  18);
        await token.connect(addr1).approve(bridge.address, amount);
        await bridge.lockAssets(token.address, amount, "Ethereum");

        await expect(bridge.connect(addr1).withdrawLockedAssets(token.address, amount))
            .to.be.revertedWith("Only the owner can withdraw locked assets");
    });
});
