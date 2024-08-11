// oracle.js
const Web3 = require('web3');
const axios = require('axios');

class Oracle {
    constructor() {
        this.web3 = new Web3(new Web3.providers.HttpProvider('https://pi-network-node.com'));
        this.oracleAddress = '0x...'; // Oracle service contract address
    }

    async getMarketPrice() {
        // Call external API to get current market price of PSI
        const response = await axios.get('https://api.coingecko.com/api/v3/simple/price?ids=pi-stable-coin&vs_currencies=usd');
        return response.data.pi_stable_coin.usd;
    }

    async updatePriceFeed() {
        // Update price feed on the blockchain
        const price = await this.getMarketPrice();
        this.web3.eth.Contract(this.oracleAddress, 'updatePriceFeed', price);
    }
}

module.exports = Oracle;
