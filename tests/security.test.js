// tests/security.test.js
const Security = require('../security'); // Assuming you have a security module

describe('Security Module', () => {
    test('should encrypt data correctly', () => {
        const data = 'sensitive data';
        const encryptedData = Security.encrypt(data);
        expect(encryptedData).not.toBe(data);
    });

    test('should decrypt data correctly', () => {
        const data = 'sensitive data';
        const encryptedData = Security.encrypt(data);
        const decryptedData = Security.decrypt(encryptedData);
        expect(decryptedData).toBe(data);
    });
});
