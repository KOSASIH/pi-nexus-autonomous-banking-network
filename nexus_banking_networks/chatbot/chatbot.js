// chatbot.js
const natural = require("natural");
const tokenizer = new natural.WordTokenizer();

class Chatbot {
  constructor() {
    this.intents = {
      balance: this.handleBalance,
      transfer: this.handleTransfer,
      help: this.handleHelp,
    };
  }

  processMessage(message) {
    const tokens = tokenizer.tokenize(message);
    const intent = this.determineIntent(tokens);
    if (intent) {
      return this.intents[intent](tokens);
    } else {
      return "I didn't understand that. Please try again!";
    }
  }

  determineIntent(tokens) {
    // Implement intent determination logic here
    // For example, using a simple keyword-based approach
    if (tokens.includes("balance")) {
      return "balance";
    } else if (tokens.includes("transfer")) {
      return "transfer";
    } else if (tokens.includes("help")) {
      return "help";
    } else {
      return null;
    }
  }

  handleBalance(tokens) {
    // Implement balance handling logic here
    return "Your current balance is $1000.";
  }

  handleTransfer(tokens) {
    // Implement transfer handling logic here
    return "Transfer successful!";
  }

  handleHelp(tokens) {
    // Implement help handling logic here
    return "I can help you with balance inquiries and transfers. What would you like to do?";
  }
}

// Example usage:
const chatbot = new Chatbot();
const message = "What is my balance?";
console.log(chatbot.processMessage(message)); // Output: Your current balance is $1000.
