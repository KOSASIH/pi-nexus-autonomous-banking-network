// anomalyDetection.js

class AnomalyDetection {
    constructor() {
        this.transactionHistory = [];
        this.threshold = {
            amount: 1000, // Example threshold for transaction amount
            frequency: 5,  // Example threshold for transaction frequency
        };
    }

    // Log a transaction
    logTransaction(userId, amount) {
        const transaction = {
            userId,
            amount,
            timestamp: new Date(),
        };
        this.transactionHistory.push(transaction);
        console.log(`Transaction logged: ${JSON.stringify(transaction)}`);
    }

    // Check for anomalies in the transaction history
    checkForAnomalies(userId) {
        const userTransactions = this.transactionHistory.filter(tx => tx.userId === userId);
        const totalAmount = userTransactions.reduce((sum, tx) => sum + tx.amount, 0);
        const transactionCount = userTransactions.length;

        const anomalies = [];
        if (totalAmount > this.threshold.amount) {
            anomalies.push(`High transaction amount: $${totalAmount}`);
        }
        if (transactionCount > this.threshold.frequency) {
            anomalies.push(`High transaction frequency: ${transactionCount} transactions`);
        }

        return anomalies;
    }
}

// Example usage
(async () => {
    const anomalyDetector = new AnomalyDetection();
    const userId = 'user123';

    // Simulate logging transactions
    anomalyDetector.logTransaction(userId, 500);
    anomalyDetector.logTransaction(userId, 600);
    anomalyDetector.logTransaction(userId, 200);
    anomalyDetector.logTransaction(userId, 300);
    anomalyDetector.logTransaction(userId, 1500); // This should trigger an anomaly

 // Check for anomalies
    const anomalies = anomalyDetector.checkForAnomalies(userId);
    if (anomalies.length > 0) {
        console.log(`Anomalies detected for user ${userId}: ${anomalies.join(', ')}`);
    } else {
        console.log(`No anomalies detected for user ${userId}.`);
    }
})();

export default AnomalyDetection;
       
