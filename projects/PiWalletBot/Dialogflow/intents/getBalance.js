import { WebhookClient } from 'dialogflow-fulfillment';
import { WalletService } from '../services/walletService';

const walletService = new WalletService();

export async function getBalance(agent) {
  const userId = agent.session.userId;
  try {
    const balance = await walletService.getBalance(userId);
    agent.add(`Your current balance is ${balance} PIC`);
  } catch (error) {
    agent.add(`Sorry, I'm having trouble retrieving your balance. Please try again later.`);
  }
}

export let client = new WebhookClient({ request: req, response: res });
