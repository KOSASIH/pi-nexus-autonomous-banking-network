// cryptoPaymentGateway.js

class CryptoPaymentGateway {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.baseUrl = 'https://api.crypto-payment-gateway.com';
    }

    // Process a payment
    async processPayment(amount, currency, paymentDetails) {
        const response = await fetch(`${this.baseUrl}/process`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                amount,
                currency,
                paymentDetails,
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to process payment');
        }

        return await response.json();
    }

    // Handle payment callback from the gateway
    async handlePaymentCallback(callbackData) {
        // Here you would implement logic to handle the callback
        // For example, verifying the payment status and updating your database
        console.log('Payment callback received:', callbackData);
        // Process the callback data as needed
    }

    // Get transaction details
    async getTransactionDetails(transactionId) {
        const response = await fetch(`${this.baseUrl}/transaction/${transactionId}`, {
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
const paymentGateway = new CryptoPaymentGateway({ apiKey: 'YOUR_API_KEY' });
paymentGateway.processPayment(0.1, 'BTC', { recipient: 'recipient_address' })
    .then(result => console.log('Payment processed:', result))
    .catch(error => console.error('Error processing payment:', error));
