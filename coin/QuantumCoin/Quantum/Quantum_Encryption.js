// Quantum/Quantum_Encryption.js
const crypto = require('crypto');

// Function to generate a random quantum key
function generateQuantumKey(length) {
    const key = crypto.randomBytes(length).toString('base64');
    return key;
}

// Function to simulate quantum key distribution
function distributeQuantumKey(senderKey, receiverKey) {
    // Simulate the process of sending the key
    console.log("Sender's Key: ", senderKey);
    console.log("Receiver's Key: ", receiverKey);

    // Simulate eavesdropping detection
    const eavesdropperDetected = Math.random() < 0.1; // 10% chance of eavesdropping
    if (eavesdropperDetected) {
        console.log("Eavesdropping detected! Key distribution failed.");
        return null;
    }

    console.log("Key distribution successful!");
    return senderKey; // Return the key if no eavesdropping is detected
}

// Function to encrypt a message using the quantum key
function encryptMessage(message, key) {
    const buffer = Buffer.from(message, 'utf-8');
    const encrypted = Buffer.concat([buffer, Buffer.from(key)]);
    return encrypted.toString('base64');
}

// Function to decrypt a message using the quantum key
function decryptMessage(encryptedMessage, key) {
    const buffer = Buffer.from(encryptedMessage, 'base64');
    const decrypted = buffer.slice(0, buffer.length - key.length);
    return decrypted.toString('utf-8');
}

// Example usage
const quantumKeyLength = 16; // Length of the quantum key
const senderKey = generateQuantumKey(quantumKeyLength);
const receiverKey = generateQuantumKey(quantumKeyLength);

const distributedKey = distributeQuantumKey(senderKey, receiverKey);
if (distributedKey) {
    const message = "Hello, Quantum World!";
    const encryptedMessage = encryptMessage(message, distributedKey);
    console.log("Encrypted Message: ", encryptedMessage);

    const decryptedMessage = decryptMessage(encryptedMessage, distributedKey);
    console.log("Decrypted Message: ", decryptedMessage);
}
