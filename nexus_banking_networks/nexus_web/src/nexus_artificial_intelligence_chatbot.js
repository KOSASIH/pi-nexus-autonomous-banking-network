import { Chatbot } from 'rivescript';

class NexusArtificialIntelligenceChatbot {
    constructor() {
        this.bot = new Chatbot();
        this.bot.load('nexus_chatbot.rive');
    }

    respond(input) {
        return this.bot.reply('localuser', input);
    }
}

const chatbot = new NexusArtificialIntelligenceChatbot();

// Example usage:
console.log(chatbot.respond('Hello, how are you?'));
