const avalanche = require("avalanche-js");
const Web3 = require("web3");
const AvalancheBridgeContract = require("./AvalancheBridge.sol");

class AvalancheBridge {
    constructor(avalancheAddress, piNetworkAddress, avalancheBridgeContractAddress) {
        this.avalancheAddress = avalancheAddress;
        this.piNetworkAddress = piNetworkAddress;
        this.avalancheBridgeContractAddress = avalancheBridgeContractAddress;
        this.avalanche = avalanche;
        this.web3 = new Web3(new Web3.providers.HttpProvider("https://api.avax.network/ext/bc/C/rpc"));
    }

    async transferTokens(token, amount) {
        const txCount = await this.web3.eth.getTransactionCount(this.avalancheAddress);
        const tx = {
            from: this.avalancheAddress,
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
        const balance = await this.avalanche.getBalance(this.avalancheAddress, token);
        return balance;
    }

    async getAvalancheBridgeContract() {
        const contract = new this.web3.eth.Contract(AvalancheBridgeContract.abi, this.avalancheBridgeContractAddress);
        return contract;
    }
}

module.exports = AvalancheBridge;
