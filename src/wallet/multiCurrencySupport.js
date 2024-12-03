// multiCurrencySupport.js

class MultiCurrencySupport {
    constructor() {
        this.currencies = {
            USD: 1, // Base currency
            EUR: 0.85,
            GBP: 0.75,
            BTC: 0.000025, // Example conversion rate for Bitcoin
            ETH: 0.0004,   // Example conversion rate for Ethereum
        };
    }

    // Convert an amount from one currency to another
    convert(amount, fromCurrency, toCurrency) {
        if (!this.currencies[fromCurrency] || !this.currencies[toCurrency]) {
            throw new Error("Unsupported currency.");
        }
        const baseAmount = amount / this.currencies[fromCurrency];
        return baseAmount * this.currencies[toCurrency];
    }

    // Get available currencies
    getAvailableCurrencies() {
        return Object.keys(this.currencies);
    }

    // Get conversion rate between two currencies
    getConversionRate(fromCurrency, toCurrency) {
        if (!this.currencies[fromCurrency] || !this.currencies[toCurrency]) {
            throw new Error("Unsupported currency.");
        }
        return this.currencies[toCurrency] / this.currencies[fromCurrency];
    }
}

// Example usage
const currencySupport = new MultiCurrencySupport();
const amountInUSD = 100;
const amountInEUR = currencySupport.convert(amountInUSD, 'USD', 'EUR');
console.log(`$${amountInUSD} is equivalent to €${amountInEUR.toFixed(2)}`);

const availableCurrencies = currencySupport.getAvailableCurrencies();
console.log("Available currencies:", availableCurrencies.join(', '));

const conversionRate = currencySupport.getConversionRate('USD', 'BTC');
console.log(`Conversion rate from USD to BTC: ${conversionRate}`);

export default MultiCurrencySupport;
