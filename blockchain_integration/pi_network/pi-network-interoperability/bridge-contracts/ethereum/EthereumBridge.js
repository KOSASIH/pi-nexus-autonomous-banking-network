const Web3 = require("web3");
const ethereumBridgeContract = require("./EthereumBridge.sol");

class EthereumBridge {
    constructor(ethereumAddress, piNetworkAddress) {
        this.ethereumAddress = ethereumAddress;
        this.piNetworkAddress = piNetworkAddress;
        this.web3 = new Web3(new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"));
    }

    async transferTokens(token, amount) {
        const txCount = await this.web3.eth.getTransactionCount(this.ethereumAddress);
        const tx = {
            from: this.ethereumAddress,
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
        const balance = await this.web3.eth.getBalance(this.ethereumAddress);
        return balance;
    }
}

module.exports = EthereumBridge;
