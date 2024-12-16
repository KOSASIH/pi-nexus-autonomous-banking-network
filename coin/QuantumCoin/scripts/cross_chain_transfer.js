// cross_chain_transfer.js
const { ethers } = require("ethers");
const CrossChainBridgeABI = require("./artifacts/CrossChainBridge.json"); // Adjust the path as necessary
const ERC20ABI = require("./artifacts/ERC20.json"); // Adjust the path as necessary

const provider = new ethers.providers.JsonRpcProvider("YOUR_INFURA_OR_ALCHEMY_URL");
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);
const bridgeContractAddress = "YOUR_CROSS_CHAIN_BRIDGE_ADDRESS";
const tokenAddress = "YOUR_ERC20_TOKEN_ADDRESS"; // The token to be locked

async function lockAssets(amount, destinationChain) {
    const bridgeContract = new ethers.Contract(bridgeContractAddress, CrossChainBridgeABI.abi, wallet);
    const tokenContract = new ethers.Contract(tokenAddress, ERC20ABI.abi, wallet);

    try {
        // Approve the bridge contract to spend tokens
        const approvalTx = await tokenContract.approve(bridgeContractAddress, amount);
        console.log(`Approving tokens... Transaction Hash: ${approvalTx.hash}`);
        await approvalTx.wait();
        console.log("Tokens approved!");

        // Lock the assets
        const lockTx = await bridgeContract.lockAssets(tokenAddress, amount, destinationChain);
        console.log(`Locking assets... Transaction Hash: ${lockTx.hash}`);
        await lockTx.wait();
        console.log("Assets locked successfully!");
    } catch (error) {
        console.error("Error locking assets:", error);
    }
}

// Example usage
const amountToLock = ethers.utils.parseUnits("10", 18); // Amount to lock (10 tokens)
const destinationChain = "Ethereum"; // Replace with the actual destination chain name

lockAssets(amountToLock, destinationChain);
