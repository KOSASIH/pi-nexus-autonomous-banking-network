const walletUrl = 'https://minepi.com';
const nodeUrl = 'https://node1.pi.network';

const wallet = {
  async getWallet() {
    const response = await fetch(`${walletUrl}/api/v1/wallet`);
    return response.json();
  },

  async sendTransaction() {
    const response = await fetch(`${walletUrl}/api/v1/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        // ...
      }),
    });
    return response.json();
  },

  // ...
};

module.exports = wallet;
