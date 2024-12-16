// Quantum_Secure_Messaging.js

const crypto = require('crypto');
const { QuantumKeyDistribution } = require('./QuantumKeyDistribution');

class QuantumSecureMessaging {
    constructor() {
        this.qkd = new QuantumKeyDistribution();
        this.sharedKey = null;
    }

    // Generate a quantum key
    generateQuantumKey() {
        this.sharedKey = this.qkd.generateKey();
        console.log('Quantum key generated:', this.sharedKey);
    }

    // Encrypt a message using the shared key
    encryptMessage(message) {
        if (!this.sharedKey) {
            throw new Error('Shared key not generated. Call generateQuantumKey() first.');
        }
        const cipher = crypto.createCipheriv('aes-256-cbc', this.sharedKey, this.sharedKey.slice(0, 16));
        let encrypted = cipher.update(message, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        return encrypted;
    }

    // Decrypt a message using the shared key
    decryptMessage(encryptedMessage) {
        if (!this.sharedKey) {
            throw new Error('Shared key not generated. Call generateQuantumKey() first.');
        }
        const decipher = crypto.createDecipheriv('aes-256-cbc', this.sharedKey, this.sharedKey.slice(0, 16));
        let decrypted = decipher.update(encryptedMessage, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    }
}

// Example usage
const messaging = new QuantumSecureMessaging();
messaging.generateQuantumKey();
const message = "Hello, this is a secure message!";
const encryptedMessage = messaging.encryptMessage(message);
console.log('Encrypted Message:', encryptedMessage);
const decryptedMessage = messaging.decryptMessage(encryptedMessage);
console.log('Decrypted Message:', decryptedMessage);
