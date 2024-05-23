const Web3 = require('web3');
const Escrow = require('../contracts/templates/Escrow.json');
const Voting = require('../contracts/templates/Voting.json');
const Crowdfunding = require('../contracts/templates/Crowdfunding.json');

class SmartContractTemplateService {

    constructor(web3, escrowAddress, votingAddress, crowdfundingAddress) {
        this.web3 = web3;
        this.escrow = new this.web3.eth.Contract(Escrow.abi, escrowAddress);
        this.voting = new this.web3.eth.Contract(Voting.abi, votingAddress);
        this.crowdfunding = new this.web3.eth.Contract(Crowdfunding.abi, crowdfundingAddress);
    }

    // The function to release an escrowed asset
    releaseEscrow() {
        return new Promise((resolve, reject) => {
            this.escrow.methods.release().send({ from: this.web3.eth.defaultAccount })
                .on('receipt', (receipt) => {
                    resolve(receipt);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }

    // The function to add a candidate
    addCandidate(name) {
        return new Promise((resolve, reject) => {
            this.voting.methods.addCandidate(name).send({ from: this.web3.eth.defaultAccount })
                .on('receipt', (receipt) => {
                    resolve(receipt);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }

    // The function to vote for a candidate
    vote(index) {
        return new Promise((resolve, reject) => {
            this.voting.methods.vote(index).send({ from: this.web3.eth.defaultAccount })
                .on('receipt', (receipt) => {
                    resolve(receipt);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }

    // The function to contribute to a crowdfunding campaign
    contribute(amount) {
        return new Promise((resolve, reject) => {
            this.crowdfunding.methods.contribute().send({ from: this.web3.eth.defaultAccount, value: this.web3.utils.toWei(amount.toString(), 'ether') })
                .on('receipt', (receipt) => {
                    resolve(receipt);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }

    // The function to refund a contribution
    refund() {
        return new Promise((resolve, reject) => {
            this.crowdfunding.methods.refund().send({ from: this.web3.eth.defaultAccount })
                .on('receipt', (receipt) => {
                    resolve(receipt);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }

}

module.exports = SmartContractTemplateService;
