// oracles/priceFeed.js

const { ethers } = require("ethers");
const { ChainlinkPriceFeedABI } = require("./abi/ChainlinkPriceFeedABI.json");

// Replace with the appropriate network and price feed address
const NETWORK = "mainnet"; // or "rinkeby", "ropsten", etc.
const PRICE_FEED_ADDRESS = "0x5f4eC3Df9cbd43714fe2740f5e3616155c5B8419"; // ETH/USD price feed on mainnet

class PriceFeed {
    constructor() {
        this.provider = new ethers.providers.InfuraProvider(NETWORK, "YOUR_INFURA_PROJECT_ID");
        this.priceFeedContract = new ethers.Contract(PRICE_FEED_ADDRESS, ChainlinkPriceFeedABI, this.provider);
    }

    async getLatestPrice() {
        try {
            const price = await this.priceFeedContract.latestRoundData();
            const latestPrice = price[1]; // price[1] contains the price
            return ethers.utils.formatUnits(latestPrice, 8); // Chainlink price feeds return 8 decimal places
        } catch (error) {
            console.error("Error fetching price:", error);
            throw error;
        }
    }
}

module.exports = PriceFeed;
