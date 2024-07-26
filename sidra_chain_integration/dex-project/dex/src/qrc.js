// qrc.js
const liboqs = require('liboqs');

// Set up the quantum-resistant cryptography algorithm
const algorithm = 'FrodoKEM'; // Choose a supported algorithm from liboqs
const kem = new liboqs.KEM(algorithm);

// Generate a key pair
const { publicKey, privateKey } = kem.generateKeyPair();

// Encrypt a message using the public key
const message = 'Hello, Quantum World!';
const ciphertext = kem.encrypt(publicKey, message);

// Decrypt the message using the private key
const decryptedMessage = kem.decrypt(privateKey, ciphertext);

console.log(`Original message: ${message}`);
console.log(`Decrypted message: ${decryptedMessage}`);
