// marketDataAPI.js
const axios = require('axios');

// Example function to fetch current interest rates from a market data API
async function fetchCurrentInterestRates() {
    try {
        const response = await axios.get('https://api.example.com/interest-rates'); // Replace with actual API endpoint
        return response.data; // Assuming the API returns data in a usable format
    } catch (error) {
        console.error('Error fetching interest rates:', error.message);
        throw new Error('Failed to fetch interest rates');
    }
}

// Example function to fetch currency exchange rates
async function fetchExchangeRates(baseCurrency = 'USD') {
    try {
        const response = await axios.get(`https://api.exchangerate-api.com/v4/latest/${baseCurrency}`); // Replace with actual API endpoint
        return response.data.rates; // Assuming the API returns rates in a usable format
    } catch (error) {
        console.error('Error fetching exchange rates:', error.message);
        throw new Error('Failed to fetch exchange rates');
    }
}

// Exporting functions for use in other modules
module.exports = {
    fetchCurrentInterestRates,
    fetchExchangeRates
};
