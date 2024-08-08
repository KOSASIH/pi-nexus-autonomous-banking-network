import { WebhookClient } from 'dialogflow-fulfillment';
import { WalletService } from '../services/walletService';

const walletService = new WalletService();

export async function sendTransaction(agent) {
  const userId = agent.session.userId;
  const to = agent.parameters.to;
  const value = agent.parameters.value;

  try {
    await walletService.sendTransaction(userId, to, value);
    agent.add(`Transaction sent successfully!`);
  } catch (error) {
    agent.add(`Sorry, I'm having trouble sending the transaction. Please try again later.`);
  }
}

export let client = new WebhookClient({ request: req, response: res });
