import Agent from './Agent';

class TransactionMonitoringAgent extends Agent {
    constructor() {
        super('TransactionMonitoringAgent');
    }

    monitorTransaction(transaction) {
        // Logic to monitor transaction
        if (transaction.amount > 10000) { // Example threshold
            this.sendAlert(transaction);
        }
    }

    sendAlert(transaction) {
        // Logic to send alert (e.g., email, SMS)
        this.log(`Alert: High transaction detected - ${JSON.stringify(transaction)}`);
    }
}

export default TransactionMonitoringAgent;
