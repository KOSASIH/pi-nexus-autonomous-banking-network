const Web3 = require('web3');
const PiToEthBridge = require('../contracts/PiToEthBridge.json');

class PiToEthBridgeService {

    constructor(web3, contractAddress) {
        this.web3 = web3;
        this.contract = new this.web3.eth.Contract(PiToEthBridge.abi, contractAddress);
    }

    // The function to convert PI tokens to Ethereum tokens
    convertPiToEthereum(amount) {
        return new Promise((resolve, reject) => {
            this.contract.methods.convertPiToEthereum(amount).send({
                from: this.web3.eth.defaultAccount
            }, (error, result) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(result);
                }
            });
        });
    }

    // The function to convert Ethereum tokens to PI tokens
    convertEthereumToPi(amount) {
        return new Promise((resolve, reject) => {
            this.contract.methods.convertEthereumToPi(amount).send({
                from: this.web3.eth.defaultAccount
            }, (error, result) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(result);
                }
            });
        });
    }

}

module.exports = PiToEthBridgeService;
