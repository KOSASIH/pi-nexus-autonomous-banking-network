// multiCurrencySupport.test.js

import MultiCurrencySupport from './multiCurrencySupport'; // Assuming you have a MultiCurrencySupport class/module

describe('Multi-Currency Support', () => {
    let currencySupport;

    beforeEach(() => {
        currencySupport = new MultiCurrencySupport();
    });

    test('should convert currency correctly', () => {
        const amount = 100;
        const fromCurrency = 'USD';
        const toCurrency = 'EUR';
        const result = currencySupport.convert(amount, fromCurrency, toCurrency);
        expect(result).toBeCloseTo(85, 2); // Assuming 1 USD = 0.85 EUR
    });

    test('should throw error for unsupported currency', () => {
        expect(() => currencySupport.convert(100, 'USD', 'XYZ')).toThrow('Currency XYZ is not supported');
    });

    test('should return the same amount for the same currency', () => {
        const result = currencySupport.convert(100, 'USD', 'USD');
        expect(result).toBe(100);
    });
});
