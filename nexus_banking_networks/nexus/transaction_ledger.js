const { Blockchain, Block } = require('./blockchain');

class TransactionLedger {
    constructor() {
        this.blockchain = new Blockchain();
    }

    addTransaction(transaction) {
        const block = new Block(transaction);
        this.blockchain.addBlock(block);
    }

    getTransactionHistory() {
        return this.blockchain.getBlocks();
    }

    validateTransaction(transaction) {
        // Implement transaction validation logic using blockchain
    }
}

module.exports = TransactionLedger;
