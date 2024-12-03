// biometricAuth.test.js

import BiometricAuth from './biometricAuth'; // Assuming you have a BiometricAuth class/module

describe('Biometric Authentication', () => {
    let biometricAuth;

    beforeEach(() => {
        biometricAuth = new BiometricAuth();
    });

    test('should successfully authenticate with valid fingerprint', () => {
        const result = biometricAuth.authenticate('validFingerprint');
        expect(result).toBe(true);
    });

    test('should fail authentication with invalid fingerprint', () => {
        const result = biometricAuth.authenticate('invalidFingerprint');
        expect(result).toBe(false);
    });

    test('should throw error if fingerprint is not provided', () => {
        expect(() => biometricAuth.authenticate()).toThrow('Fingerprint is required');
    });

    test('should return false if biometric data is not enrolled', () => {
        biometricAuth.enroll('validFingerprint');
        const result = biometricAuth.authenticate('notEnrolledFingerprint');
        expect(result).toBe(false);
    });
});
