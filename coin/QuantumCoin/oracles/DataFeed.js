// scripts/oracles/DataFeed.js
const { ethers } = require("hardhat");

async function main() {
    // Replace with your deployed PriceOracle contract address
    const priceOracleAddress = "0xYourPriceOracleAddress"; // Update with actual address
    const PriceOracle = await ethers.getContractAt("PriceOracle", priceOracleAddress);

    // Function to update price
    async function updatePrice(asset, price) {
        const tx = await PriceOracle.updatePrice(asset, ethers.utils.parseEther(price.toString()));
        await tx.wait();
        console.log(`Price of ${asset} updated to ${price}`);
    }

    // Function to get price
    async function getPrice(asset) {
        const price = await PriceOracle.getPrice(asset);
        console.log(`Current price of ${asset}: ${ethers.utils.formatEther(price)} ETH`);
    }

    // Example usage
    await updatePrice("ETH", 3000); // Update the price of ETH
    await getPrice("ETH"); // Get the current price of ETH
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
