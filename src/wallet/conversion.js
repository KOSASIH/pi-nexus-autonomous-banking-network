// wallet/conversion.js
const axios = require('axios');

class CurrencyConverter {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.apiUrl = 'https://api.exchangerate-api.com/v4/latest/';
    }

    async convert(amount, fromCurrency, toCurrency) {
        try {
            const response = await axios.get(`${this.apiUrl}${fromCurrency}`);
            const rates = response.data.rates;
            if (!rates[toCurrency]) throw new Error('Invalid currency code');
            const convertedAmount = (amount * rates[toCurrency]) / rates[fromCurrency];
            return convertedAmount;
        } catch (error) {
            throw new Error(`Currency conversion failed: ${error.message}`);
        }
    }
}

module.exports = CurrencyConverter;
