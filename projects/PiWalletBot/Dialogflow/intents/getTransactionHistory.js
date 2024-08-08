import { WebhookClient } from 'dialogflow-fulfillment';
import { WalletService } from '../services/walletService';

const walletService = new WalletService();

export async function getTransactionHistory(agent) {
  const userId = agent.session.userId;

  try {
    const transactions = await walletService.getTransactionHistory(userId);
    agent.add(`Here is your transaction history:`);
    transactions.forEach((transaction) => {
      agent.add(`  - ${transaction.date}: ${transaction.type} ${transaction.value} PIC to ${transaction.to}`);
    });
  } catch (error) {
    agent.add(`Sorry, I'm having trouble retrieving your transaction history. Please try again later.`);
  }
}

export let client = new WebhookClient({ request: req, response: res });
