// transactionHistory.js

class TransactionHistory {
    constructor() {
        this.transactions = [];
    }

    // Add a transaction
    addTransaction(transaction) {
        this.transactions.push(transaction);
        console.log(`Transaction added: ${JSON.stringify(transaction)}`);
    }

    // Filter transactions based on criteria
    filterTransactions(criteria) {
        return this.transactions.filter(transaction => {
            const matchesDate = criteria.startDate ? new Date(transaction.date) >= new Date(criteria.startDate) : true;
            const matchesEndDate = criteria.endDate ? new Date(transaction.date) <= new Date(criteria.endDate) : true;
            const matchesAmount = criteria.minAmount ? transaction.amount >= criteria.minAmount : true;
            const matchesCurrency = criteria.currency ? transaction.currency === criteria.currency : true;

            return matchesDate && matchesEndDate && matchesAmount && matchesCurrency;
        });
    }

    // Get all transactions
    getAllTransactions() {
        return this.transactions;
    }
}

// Example usage
const transactionHistory = new TransactionHistory();

// Adding some transactions
transactionHistory.addTransaction({ id: 1, amount: 200, date: '2023-10-01', currency: 'USD' });
transactionHistory.addTransaction({ id: 2, amount: 150, date: '2023-10-02', currency: 'EUR' });
transactionHistory.addTransaction({ id: 3, amount: 0.01, date: '2023-10-03', currency: 'BTC' });

// Filtering transactions
const filteredTransactions = transactionHistory.filterTransactions({
    startDate: '2023-10-01',
    endDate: '2023-10-02',
    minAmount: 100,
    currency: 'USD',
});

console.log("Filtered Transactions:", filteredTransactions);

export default TransactionHistory;
