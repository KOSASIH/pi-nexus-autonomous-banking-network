const apiUrl = 'https://minepi.com';
const nodeUrl = 'https://node1.pi.network';
const walletUrl = 'https://wallet.pi.network';

const api = {
  async getBalance() {
    const response = await fetch(`${apiUrl}/api/v1/balance`);
    return response.json();
  },

  async getTransactions() {
    const response = await fetch(`${apiUrl}/api/v1/transactions`);
    return response.json();
  },

  // ...
};

module.exports = api;
