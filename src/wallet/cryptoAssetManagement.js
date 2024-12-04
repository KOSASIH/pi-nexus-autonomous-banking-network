// cryptoAssetManagement.js

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const crypto = require('crypto');

// Configuration for external API
const COINGECKO_API_URL = 'https://api.coingecko.com/api/v3';
const ASSET_STORAGE = {}; // In-memory storage for user assets

class CryptoAssetManagement {
    constructor(userId) {
        this.userId = userId;
        this.assets = ASSET_STORAGE[userId] || {};
    }

    // Add a new asset to the user's portfolio
    addAsset(symbol, amount, purchasePrice) {
        const assetId = uuidv4();
        const timestamp = new Date().toISOString();
        
        this.assets[assetId] = {
            symbol,
            amount,
            purchasePrice,
            timestamp,
            currentPrice: purchasePrice,
            value: amount * purchasePrice,
        };

        ASSET_STORAGE[this.userId] = this.assets;
        return this.assets[assetId];
    }

    // Remove an asset from the user's portfolio
    removeAsset(assetId) {
        if (this.assets[assetId]) {
            delete this.assets[assetId];
            ASSET_STORAGE[this.userId] = this.assets;
            return true;
        }
        return false;
    }

    // Update the amount of an existing asset
    updateAsset(assetId, newAmount) {
        if (this.assets[assetId]) {
            this.assets[assetId].amount = newAmount;
            this.assets[assetId].value = newAmount * this.assets[assetId].currentPrice;
            ASSET_STORAGE[this.userId] = this.assets;
            return this.assets[assetId];
        }
        return null;
    }

    // Fetch current price of a specific asset
    async fetchCurrentPrice(symbol) {
        try {
            const response = await axios.get(`${COINGECKO_API_URL}/simple/price?ids=${symbol}&vs_currencies=usd`);
            return response.data[symbol].usd;
        } catch (error) {
            console.error('Error fetching current price:', error);
            throw new Error('Could not fetch current price');
        }
    }

    // Update current prices for all assets in the portfolio
    async updateCurrentPrices() {
        for (const assetId in this.assets) {
            const asset = this.assets[assetId];
            asset.currentPrice = await this.fetchCurrentPrice(asset.symbol);
            asset.value = asset.amount * asset.currentPrice;
        }
        ASSET_STORAGE[this.userId] = this.assets;
    }

    // Get the total value of the user's portfolio
    getTotalPortfolioValue() {
        return Object.values(this.assets).reduce((total, asset) => total + asset.value, 0);
    }

    // Get detailed portfolio information
    getPortfolioDetails() {
        return {
            userId: this.userId,
            assets: this.assets,
            totalValue: this.getTotalPortfolioValue(),
        };
    }

    // Generate a report of the user's portfolio
    generatePortfolioReport() {
        const report = this.getPortfolioDetails();
        console.log('Portfolio Report:', JSON.stringify(report, null, 2));
        return report;
    }
}

// Example usage
(async () => {
    const userId = 'user123';
    const cryptoManager = new CryptoAssetManagement(userId);

    // Adding assets
    cryptoManager.addAsset('bitcoin', 0.5, 30000);
    cryptoManager.addAsset('ethereum', 2, 2000);

    // Update current prices
    await cryptoManager.updateCurrentPrices();

    // Generate report
    cryptoManager.generatePortfolioReport();

    // Remove an asset
    cryptoManager.removeAsset(Object.keys(cryptoManager.assets)[0]);

    // Final report
    cryptoManager.generatePortfolioReport();
})();

module.exports = CryptoAssetManagement;
