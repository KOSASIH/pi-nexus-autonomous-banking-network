// secureMessaging.test.js

const SecureMessaging = require('./secureMessaging'); // Assuming you have a SecureMessaging module

describe('Secure Messaging Features', () => {
    let messaging;

    beforeEach(() => {
        messaging = new SecureMessaging();
    });

    test('should send a message securely', () => {
        const result = messaging.sendMessage('user1', 'user2', 'Hello, World!');
        expect(result).toBe(true);
    });

    test('should receive a message securely', () => {
        messaging.sendMessage('user1', 'user2', 'Hello, World!');
        const messages = messaging.receiveMessages('user2');
        expect(messages).toContainEqual(expect.objectContaining({
            from: 'user1',
            content: 'Hello, World!',
        }));
    });

    test('should encrypt messages before sending', () => {
        const message = 'Secret Message';
        const encryptedMessage = messaging.encryptMessage(message);
        expect(encryptedMessage).not.toBe(message);
    });

    test('should decrypt messages after receiving', () => {
        const message = 'Secret Message';
        const encryptedMessage = messaging.encryptMessage(message);
        const decryptedMessage = messaging.decryptMessage(encryptedMessage);
        expect(decryptedMessage).toBe(message);
    });
});
