const Web3 = require("web3");
const binanceSmartChainBridgeContract = require("./BinanceSmartChainBridge.sol");

class BinanceSmartChainBridge {
    constructor(binanceSmartChainAddress, piNetworkAddress) {
        this.binanceSmartChainAddress = binanceSmartChainAddress;
        this.piNetworkAddress = piNetworkAddress;
        this.web3 = new Web3(new Web3.providers.HttpProvider("https://bsc-dataseed.binance.org/api/v1/bc/BSC/main"));
    }

    async transferTokens(token, amount) {
        const txCount = await this.web3.eth.getTransactionCount(this.binanceSmartChainAddress);
        const tx = {
            from: this.binanceSmartChainAddress,
            to: this.piNetworkAddress,
            value: amount,
            gas: "20000",
            gasPrice: "20.0",
            nonce: txCount
        };
        const signedTx = await this.web3.eth.accounts.signTransaction(tx, "YOUR_PRIVATE_KEY");
        await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    }

    async getBalance(token) {
        const balance = await this.web3.eth.getBalance(this.binanceSmartChainAddress);
        return balance;
    }
}

module.exports = BinanceSmartChainBridge;
