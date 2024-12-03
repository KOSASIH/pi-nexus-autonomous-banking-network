// biometricAuth.js

class BiometricAuth {
    constructor() {
        this.isBiometricSupported = this.checkBiometricSupport();
    }

    // Check if the device supports WebAuthn
    checkBiometricSupport() {
        return window.PublicKeyCredential !== undefined;
    }

    // Enroll a user for biometric authentication
    async enrollBiometric(userId) {
        if (!this.isBiometricSupported) {
            throw new Error("Biometric authentication is not supported on this device.");
        }

        const publicKey = {
            challenge: new Uint8Array(32), // Random challenge
            rp: { name: "My Bank" },
            user: {
                id: new Uint8Array(16), // User ID
                name: userId,
                displayName: userId,
            },
            pubKeyCredParams: [{ type: "public-key", alg: -7 }], // ECDSA with SHA-256
            timeout: 60000,
            attestation: "direct",
        };

        try {
            const credential = await navigator.credentials.create({ publicKey });
            // Store the credential in your database
            console.log("Credential created:", credential);
            return credential;
        } catch (error) {
            console.error("Error during enrollment:", error);
            throw error;
        }
    }

    // Authenticate a user using biometric data
    async authenticate(userId) {
        if (!this.isBiometricSupported) {
            throw new Error("Biometric authentication is not supported on this device.");
        }

        const publicKey = {
            challenge: new Uint8Array(32), // Random challenge
            allowCredentials: [{
                id: new Uint8Array(32), // The ID of the stored credential
                type: "public-key",
            }],
            timeout: 60000,
        };

        try {
            const assertion = await navigator.credentials.get({ publicKey });
            // Verify the assertion with your server
            console.log("Authentication successful:", assertion);
            return assertion;
        } catch (error) {
            console.error("Error during authentication:", error);
            throw error;
        }
    }
}

// Example usage
(async () => {
    const biometricAuth = new BiometricAuth();
    const userId = 'user123';

    try {
        await biometricAuth.enrollBiometric(userId);
        const isAuthenticated = await biometricAuth.authenticate(userId);
        console.log(`User  authenticated: ${isAuthenticated}`);
    } catch (error) {
        console.error(error.message);
    }
})();

export default BiometricAuth;
