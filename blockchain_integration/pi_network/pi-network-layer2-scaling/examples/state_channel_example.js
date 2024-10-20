// state_channel_example.js
const { ethers } = require("hardhat");
const networkConfig = require("./network_config.json");

async function main() {
    const [owner, participant1, participant2] = await ethers.getSigners();
    
    // Load the ChannelManager and StateChannel contracts
    const channelManagerAddress = networkConfig.networks.rinkeby.rollupManagerAddress; // Assuming ChannelManager is deployed at this address
    const ChannelManager = await ethers.getContractFactory("ChannelManager");
    const channelManager = await ChannelManager.attach(channelManagerAddress);

    const StateChannel = await ethers.getContractFactory("StateChannel");
    const stateChannel = await StateChannel.deploy(channelManager.address, participant1.address, participant2.address);
    await stateChannel.deployed();

    // Open the channel
    console.log("Opening channel...");
    await stateChannel.connect(participant1).openChannel();
    console.log("Channel opened successfully!");

    // Update the state
    const newState = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("New State"));
    console.log("Updating state to:", newState);
    await stateChannel.connect(participant1).updateState(newState);
    console.log("State updated successfully!");

    // Get the current state
    const currentState = await stateChannel.getCurrentState();
    console.log("Current state:", currentState);

    // Close the channel
    console.log("Closing channel...");
    await stateChannel.connect(participant1).closeChannel();
    console.log("Channel closed successfully!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
