// chatbot.js
const Dialogflow = require('dialogflow');
const { v4: uuidv4 } = require('uuid');

class Chatbot {
    constructor() {
        this.dialogflow = new Dialogflow({
            credentials: {
                client_email: 'YOUR_CLIENT_EMAIL',
                private_key: 'YOUR_PRIVATE_KEY'
            }
        });
    }

    async processMessage(message) {
        const sessionId = uuidv4();
        const request = {
            session: sessionId,
            queryInput: {
                text: {
                    text: message,
                    languageCode: 'en-US'
                }
            }
        };
        const response = await this.dialogflow.sessions.detectIntent(request);
        return response.queryResult.fulfillmentText;
    }
}

// Example usage:
const chatbot = new Chatbot();
const message = 'What is my account balance?';
const response = chatbot.processMessage(message);
console.log(response);
