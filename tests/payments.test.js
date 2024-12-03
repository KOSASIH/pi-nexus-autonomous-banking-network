// tests/payments.test.js
const Payments = require('../payments'); // Assuming you have a payments module

describe('Payments Module', () => {
    let payments;

    beforeEach(() => {
        payments = new Payments();
    });

    test('should process payment correctly', () => {
        const result = payments.processPayment(100, 'credit_card');
        expect(result).toBe('Payment processed');
    });

    test('should not allow payment with insufficient funds', () => {
        expect(() => payments .processPayment(1000, 'credit_card')).toThrow('Insufficient funds');
    });
});
