// tests/loyalty.test.js
const Loyalty = require('../loyalty'); // Assuming you have a loyalty module

describe('Loyalty Module', () => {
    let loyalty;

    beforeEach(() => {
        loyalty = new Loyalty();
    });

    test('should add points correctly', () => {
        loyalty.addPoints(100);
        expect(loyalty.getPoints()).toBe(100);
    });

    test('should redeem points correctly', () => {
        loyalty.addPoints(100);
        loyalty.redeemPoints(50);
        expect(loyalty.getPoints()).toBe(50);
    });

    test('should not allow redeeming more points than available', () => {
        expect(() => loyalty.redeemPoints(50)).toThrow('Insufficient points');
    });
});
