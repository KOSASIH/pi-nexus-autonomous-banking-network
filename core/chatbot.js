const { Dialogflow } = require('dialogflow');
const { v4 as uuidv4 } = require('uuid');

class Chatbot {
  constructor() {
    this.dialogflow = new Dialogflow({
      credentials: {
        private_key: 'YOUR_PRIVATE_KEY',
        client_email: 'YOUR_CLIENT_EMAIL',
      },
    });
  }

  async processMessage(message) {
    const sessionId = uuidv4();
    const request = {
      session: sessionId,
      queryInput: {
        text: {
          text: message,
        },
      },
    };

    try {
      const response = await this.dialogflow.sessions.detectIntent(request);
      const intent = response.queryResult.intent.displayName;
      const responseText = response.queryResult.fulfillmentText;

      return { intent, responseText };
    } catch (error) {
      console.error(error);
      return { intent: 'unknown', responseText: 'Sorry, I didn\'t understand that.' };
    }
  }
}

module.exports = Chatbot;
