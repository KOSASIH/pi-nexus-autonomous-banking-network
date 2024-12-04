// blockchainAPI.js

class BlockchainAPI {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.baseUrl = 'https://api.blockchain.example.com';
    }

    // Send a transaction to the blockchain
    async sendTransaction(transaction) {
        const response = await fetch(`${this.baseUrl}/send`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transaction),
        });
        if (!response.ok) {
            throw new Error('Failed to send transaction');
        }
        return await response.json();
    }

    // Get transaction details
    async getTransactionDetails(txId) {
        const response = await fetch(`${this.baseUrl}/transaction/${txId}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
        });
        if (!response.ok) {
            throw new Error('Failed to fetch transaction details');
        }
        return await response.json();
    }
}

// Example usage
const blockchainAPI = new BlockchainAPI({ apiKey: 'YOUR_API_KEY' });
blockchainAPI.sendTransaction({ from: 'address1', to: 'address2', amount: 0.1 })
    .then(result => console.log(result));
