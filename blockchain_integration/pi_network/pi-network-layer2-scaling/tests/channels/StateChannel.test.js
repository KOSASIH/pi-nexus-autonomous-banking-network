// StateChannel.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("StateChannel Contract", function () {
    let StateChannel;
    let stateChannel;
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

        StateChannel = await ethers.getContractFactory("StateChannel");
        stateChannel = await StateChannel.deploy(channelManager.address, participant1.address, participant2.address);
        await stateChannel.deployed();
    });

    it("should open a channel successfully", async function () {
        await stateChannel.connect(participant1).openChannel();
        const isOpen = await stateChannel.isOpen();
        expect(isOpen).to.be.true;
    });

    it("should allow state updates", async function () {
        await stateChannel.connect(participant1).openChannel();
        const newState = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("New State"));
        await stateChannel.connect(participant1).updateState(newState);
        
        const currentState = await stateChannel.getCurrentState();
        expect(currentState).to.equal(newState);
    });

    it("should close the channel successfully", async function () {
        await stateChannel.connect(participant1).openChannel();
        await stateChannel.connect(participant1).closeChannel();
        
        const isOpen = await stateChannel.isOpen();
        expect(isOpen).to.be.false;
    });

    it("should revert when trying to update state after closing the channel", async function () {
        await stateChannel.connect(participant1).openChannel();
        await stateChannel.connect(participant1).closeChannel();
        
        const newState = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("New State"));
        await expect(stateChannel.connect(participant1).updateState(newState)).to.be.revertedWith("Channel is closed");
    });
});
