// tests/wallet.test.js
const Wallet = require('../wallet'); // Assuming you have a wallet module

describe('Wallet Module', () => {
    let wallet;

    beforeEach(() => {
        wallet = new Wallet();
    });

    test('should add funds correctly', () => {
        wallet.addFunds(100);
        expect(wallet.getBalance()).toBe(100);
    });

    test('should deduct funds correctly', () => {
        wallet.addFunds(100);
        wallet.deductFunds(50);
        expect(wallet.getBalance()).toBe(50);
    });

    test('should not allow overdraft', () => {
        expect(() => wallet.deductFunds(50)).toThrow('Insufficient funds');
    });
});
