// paymentAnalytics.js

class PaymentAnalytics {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.baseUrl = 'https://api.payment-analytics.com';
    }

    // Fetch payment trends over a specified period
    async fetchPaymentTrends(startDate, endDate) {
        const response = await fetch(`${this.baseUrl}/trends?start=${startDate}&end=${endDate}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Failed to fetch payment trends');
        }

        return await response.json();
    }

    // Get transaction volume for a specific currency
    async getTransactionVolume(currency) {
        const response = await fetch(`${this.baseUrl}/volume?currency=${currency}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Failed to fetch transaction volume');
        }

        return await response.json();
    }

    // Get payment success rates
    async getPaymentSuccessRates() {
        const response = await fetch(`${this.baseUrl}/success-rates`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Failed to fetch payment success rates');
        }

        return await response.json();
    }
}

// Example usage
const analytics = new PaymentAnalytics({ apiKey: 'YOUR_API_KEY' });
analytics.fetchPaymentTrends('2023-01-01', '2023-12-31')
    .then(trends => console.log('Payment trends:', trends))
    .catch(error => console.error('Error fetching payment trends:', error));
