// chatSupport.test.js

import ChatSupport from './chatSupport'; // Import the module to be tested

describe('ChatSupport', () => {
    let chatSupport;

    beforeEach(() => {
        chatSupport = new ChatSupport();
    });

    test('should initiate a chat session', () => {
        const session = chatSupport.startChat('user@example.com');
        expect(session).toHaveProperty('sessionId');
        expect(session.user).toBe('user@example.com');
    });

    test('should send a message in a chat session', () => {
        const session = chatSupport.startChat('user@example.com');
        const message = chatSupport.sendMessage(session.sessionId, 'Hello, I need help!');
        expect(message).toMatchObject({ sessionId: session.sessionId, content: 'Hello, I need help!' });
    });

    test('should receive messages in a chat session', () => {
        const session = chatSupport.startChat('user@example.com');
        chatSupport.sendMessage(session.sessionId, 'Hello, I need help!');
        const messages = chatSupport.getMessages(session.sessionId);
        expect(messages).toHaveLength(1);
        expect(messages[0].content).toBe('Hello, I need help!');
    });
});
