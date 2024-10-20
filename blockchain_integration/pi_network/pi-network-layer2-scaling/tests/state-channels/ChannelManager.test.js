// ChannelManager.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ChannelManager Contract", function () {
    let ChannelManager;
    let channelManager;
    let owner;
    let participant1;
    let participant2;

    beforeEach(async function () {
        [owner, participant1, participant2] = await ethers.getSigners();
        ChannelManager = await ethers.getContractFactory("ChannelManager");
        channelManager = await ChannelManager.deploy();
        await channelManager.deployed();
    });

    it("should create a new state channel", async function () {
        const channelAddress = await channelManager.createChannel(participant1.address, participant2.address);
        const channel = await ethers.getContractAt("StateChannel", channelAddress);
        
        const isOpen = await channel.isOpen();
        expect(isOpen).to.be.false; // Channel should be created but not opened yet
    });

    it("should allow participants to join a channel", async function () {
        const channelAddress = await channelManager.createChannel(participant1.address, participant2.address);
        const channel = await ethers.getContractAt("StateChannel", channelAddress);
        
        await channel.connect(participant1).openChannel();
        const isOpen = await channel.isOpen();
        expect(isOpen).to.be.true; // Channel should be opened by participant1
    });

    it("should revert when a non-participant tries to open a channel", async function () {
        const channelAddress = await channelManager.createChannel(participant1.address, participant2.address);
        const channel = await ethers.getContractAt("StateChannel", channelAddress);
        
        await expect(channel.connect(owner).openChannel()).to.be.revertedWith("Not a participant");
    });

    it("should allow participants to close a channel", async function () {
        const channelAddress = await channelManager.createChannel(participant1.address, participant2.address);
        const channel = await ethers.getContractAt("StateChannel", channelAddress);
        
        await channel.connect(participant1).openChannel ();
        await channel.connect(participant1).closeChannel();
        
        const isOpen = await channel.isOpen();
        expect(isOpen).to.be.false; // Channel should be closed by participant1
    });
});
