// dashboard.js

class Dashboard {
    constructor(userId) {
        this.userId = userId;
        this.transactionHistory = [];
        this.accountBalance = 0;
        this.init();
    }

    // Initialize the dashboard
    init() {
        this.render();
        this.fetchTransactionHistory();
        this.updateAccountBalance();
    }

    // Fetch transaction history (mock data)
    fetchTransactionHistory() {
        // In a real application, this would be an API call
        this.transactionHistory = [
            { id: 1, amount: 200, date: '2023-10-01', type: 'credit' },
            { id: 2, amount: 150, date: '2023-10-02', type: 'debit' },
            { id: 3, amount: 300, date: '2023-10-03', type: 'credit' },
        ];
        this.renderTransactionHistory();
    }

    // Update account balance (mock data)
    updateAccountBalance() {
        // In a real application, this would be an API call
        this.accountBalance = 1000; // Example balance
        this.renderAccountBalance();
    }

    // Render the dashboard
    render() {
        const dashboardContainer = document.createElement('div');
        dashboardContainer.id = 'dashboard';
        dashboardContainer.innerHTML = `
            <h1>User Dashboard</h1>
            <div id="account-balance"></div>
            <h2>Transaction History</h2>
            <ul id="transaction-history"></ul>
        `;
        document.body.appendChild(dashboardContainer);
    }

    // Render account balance
    renderAccountBalance() {
        const balanceElement = document.getElementById('account-balance');
        balanceElement.innerHTML = `<strong>Account Balance: $${this.accountBalance}</strong>`;
    }

    // Render transaction history
    renderTransactionHistory() {
        const historyElement = document.getElementById('transaction-history');
        historyElement.innerHTML = this.transactionHistory.map(tx => `
            <li>${tx.date}: $${tx.amount} (${tx.type})</li>
        `).join('');
    }
}

// Example usage
const userId = 'user123';
const userDashboard = new Dashboard(userId);

export default Dashboard;
