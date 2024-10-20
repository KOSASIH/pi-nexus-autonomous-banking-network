// deploy_state_channel.js
const { ethers } = require("hardhat");

async function main() {
    // Deploy ChannelManager
    const ChannelManager = await ethers.getContractFactory("ChannelManager");
    const channelManager = await ChannelManager.deploy();
    await channelManager.deployed();
    console.log("ChannelManager deployed to:", channelManager.address);

    // Deploy ChannelValidator
    const ChannelValidator = await ethers.getContractFactory("ChannelValidator");
    const channelValidator = await ChannelValidator.deploy();
    await channelValidator.deployed();
    console.log("ChannelValidator deployed to:", channelValidator.address);
}

// Execute the deployment script
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
