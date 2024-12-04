// twoFactorAuth.test.js

import TwoFactorAuth from './twoFactorAuth'; // Import the module to be tested

describe('TwoFactorAuth', () => {
    let twoFactorAuth;

    beforeEach(() => {
        twoFactorAuth = new TwoFactorAuth();
    });

    test('should generate a new 2FA code', () => {
        const code = twoFactorAuth.generateCode('user@example.com');
        expect(code).toHaveLength(6); // Assuming the code is 6 digits
    });

    test('should validate a correct 2FA code', () => {
        const code = twoFactorAuth.generateCode('user@example.com');
        const isValid = twoFactorAuth.validateCode('user@example.com', code);
        expect(isValid).toBe(true);
    });

    test('should invalidate an incorrect 2FA code', () => {
        twoFactorAuth.generateCode('user@example.com');
        const isValid = twoFactorAuth.validateCode('user@example.com', '123456');
        expect(isValid).toBe(false);
    });

    test('should expire codes after a certain time', () => {
        const code = twoFactorAuth.generateCode('user@example.com');
        jest.advanceTimersByTime(300000); // Assuming the code expires after 5 minutes
        const isValid = twoFactorAuth.validateCode('user@example.com', code);
        expect(isValid).toBe(false);
    });
});
