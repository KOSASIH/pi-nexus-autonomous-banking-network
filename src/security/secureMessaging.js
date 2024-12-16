// secureMessaging.js

class SecureMessaging {
    constructor() {
        this.users = {}; // Store user public keys
    }

    // Generate a key pair for the user
    async generateKeyPair() {
        const keyPair = await window.crypto.subtle.generateKey(
            {
                name: "RSA-OAEP",
                modulusLength: 2048,
                publicExponent: new Uint8Array([1, 0, 1]),
                hash: "SHA-256",
            },
            true,
            ["encrypt", "decrypt"]
        );
        return keyPair;
    }

    // Export the public key to share with other users
    async exportPublicKey(key) {
        const exported = await window.crypto.subtle.exportKey("spki", key);
        return btoa(String.fromCharCode(...new Uint8Array(exported)));
    }

    // Import a public key from a base64 string
    async importPublicKey(base64Key) {
        const binaryDerString = atob(base64Key);
        const binaryDer = new Uint8Array(binaryDerString.length);
        for (let i = 0; i < binaryDerString.length; i++) {
            binaryDer[i] = binaryDerString.charCodeAt(i);
        }
        return await window.crypto.subtle.importKey(
            "spki",
            binaryDer.buffer,
            {
                name: "RSA-OAEP",
                hash: "SHA-256",
            },
            true,
            ["encrypt"]
        );
    }

    // Encrypt a message using the recipient's public key
    async encryptMessage(publicKey, message) {
        const encodedMessage = new TextEncoder().encode(message);
        const encryptedMessage = await window.crypto.subtle.encrypt(
            {
                name: "RSA-OAEP",
            },
            publicKey,
            encodedMessage
        );
        return btoa(String.fromCharCode(...new Uint8Array(encryptedMessage)));
    }

    // Decrypt a message using the user's private key
    async decryptMessage(privateKey, encryptedMessage) {
        const binaryString = atob(encryptedMessage);
        const binaryArray = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            binaryArray[i] = binaryString.charCodeAt(i);
        }
        const decryptedMessage = await window.crypto.subtle.decrypt(
            {
                name: "RSA-OAEP",
            },
            privateKey,
            binaryArray.buffer
        );
        return new TextDecoder().decode(decryptedMessage);
    }

    // Send a message to a user
    async sendMessage(recipient, message) {
        const recipientPublicKey = this.users[recipient];
        if (!recipientPublicKey) {
            throw new Error("Recipient public key not found.");
        }
        const encryptedMessage = await this.encryptMessage(recipientPublicKey, message);
        // Here you would send the encrypted message to the recipient via your messaging system
        console.log(`Sending message to ${recipient}: ${encryptedMessage}`);
    }

    // Receive a message (for demonstration purposes)
    async receiveMessage(privateKey, encryptedMessage) {
        const decryptedMessage = await this.decryptMessage(privateKey, encryptedMessage);
        console.log(`Received message: ${decryptedMessage}`);
    }

    // Register a user with their public key
    async registerUser (username) {
        const keyPair = await this.generateKeyPair();
        const publicKey = await this.exportPublicKey(keyPair.publicKey);
        this.users[username] = await this.importPublicKey(publicKey);
        return keyPair.privateKey; // Return private key for decryption
    }
}

// Example usage
(async () => {
    const messaging = new SecureMessaging();
    const alicePrivateKey = await messaging.registerUser ("Alice");
    const bobPrivateKey = await messaging.registerUser ("Bob");

    // Alice sends a message to Bob
    await messaging.sendMessage("Bob", "Hello, Bob!");

    // Simulate receiving the message
    const encryptedMessage = "ENCRYPTED_MESSAGE_HERE"; // Replace with actual encrypted message
    await messaging.receiveMessage(bobPrivateKey, encryptedMessage);
})();
