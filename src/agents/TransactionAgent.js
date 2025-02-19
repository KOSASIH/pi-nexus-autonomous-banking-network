// src/agents/TransactionAgent.js
import Agent from './Agent';

class TransactionAgent extends Agent {
    constructor() {
        super('TransactionAgent');
        this.transactions = [];
    }

    processTransaction(transaction) {
        // Logic to process the transaction
        this.transactions.push(transaction);
        this.log(`Processed transaction: ${JSON.stringify(transaction)}`);
    }

    getTransactionHistory() {
        return this.transactions;
    }
}

export default TransactionAgent;
