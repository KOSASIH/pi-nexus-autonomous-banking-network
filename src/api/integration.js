// api/integration.js
const axios = require('axios');

class APIIntegration {
    constructor(baseURL) {
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 5000,
            headers: {
                'Content-Type': 'application/json',
            },
        });
    }

    async get(endpoint) {
        try {
            const response = await this.client.get(endpoint);
            return response.data;
        } catch (error) {
            this.handleError(error);
        }
    }

    async post(endpoint, data) {
        try {
            const response = await this.client.post(endpoint, data);
            return response.data;
        } catch (error) {
            this.handleError(error);
        }
    }

    handleError(error) {
        if (error.response) {
            // The request was made and the server responded with a status code
            console.error('Error Response:', error.response.data);
            throw new Error(`API Error: ${error.response.status} - ${error.response.data.message}`);
        } else if (error.request) {
            // The request was made but no response was received
            console.error('Error Request:', error.request);
            throw new Error('API Error: No response received from the server.');
        } else {
            // Something happened in setting up the request
            console.error('Error Message:', error.message);
            throw new Error(`API Error: ${error.message}`);
        }
    }
}

module.exports = APIIntegration;
