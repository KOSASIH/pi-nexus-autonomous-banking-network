// client.js
const Web3 = require('web3');
const axios = require('axios');

class PiStableCoinClient {
    constructor() {
        this.web3 = new Web3(new Web3.providers.HttpProvider('https://pi-network-node.com'));
        this.contractAddress = '0x...'; // Pi Stable Coin contract address
    }

    async getBalance(address) {
        // Get the balance of a user
        const balance = await this.web3.eth.Contract(this.contractAddress, 'balanceOf', address);
        return balance;
    }

    async transfer(from, to, amount) {
        // Transfer PSI tokens from one user to another
        const tx = await this.web3.eth.Contract(this.contractAddress, 'transfer', from, to, amount);
        return tx;
    }

    async stabilize() {
        // Call the stabilize function on the Pi Stable Coin contract
        const tx = await this.web3.eth.Contract(this.contractAddress, 'stabilize');
        return tx;
    }
}

module.exports = PiStableCoinClient;
