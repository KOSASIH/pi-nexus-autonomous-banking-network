const Web3 = require("web3");
const sidraChainTokenContract = require("./SidraChainTokenContract.sol");

class SidraChainTokenContract {
    constructor(sidraChainAddress) {
        this.sidraChainAddress = sidraChainAddress;
        this.web3 = new Web3(new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"));
    }

    async transferTokens(to, amount) {
        const txCount = await this.web3.eth.getTransactionCount(this.sidraChainAddress);
        const tx = {
            from: this.sidraChainAddress,
            to: to,
            value: amount,
            gas: "20000",
            gasPrice: "20.0",
            nonce: txCount
        };
        const signedTx = await this.web3.eth.accounts.signTransaction(tx, "YOUR_PRIVATE_KEY");
        await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    }

    async getBalance() {
        const balance = await this.web3.eth.getBalance(this.sidraChainAddress);
        return balance;
    }
}

module.exports = SidraChainTokenContract;
