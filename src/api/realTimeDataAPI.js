// realTimeDataAPI.js

class RealTimeDataAPI {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.baseUrl = 'https://api.realtime.example.com';
    }

    // Subscribe to a data feed
    async subscribeToFeed(feedName) {
        const response = await fetch(`${this.baseUrl}/subscribe`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feed: feedName }),
        });
        if (!response.ok) {
            throw new Error('Failed to subscribe to feed');
        }
        return await response.json();
    }

    // Unsubscribe from a data feed
    async unsubscribeFromFeed(feedName) {
        const response = await fetch(`${this.baseUrl}/unsubscribe`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feed: feedName }),
        });
        if (!response.ok) {
            throw new Error('Failed to unsubscribe from feed');
        }
        return await response.json();
    }
}

// Example usage
const realTimeDataAPI = new RealTimeDataAPI({ apiKey: 'YOUR_API_KEY' });
realTimeDataAPI.subscribeToFeed('live_updates')
    .then(subscription => console.log(subscription));
