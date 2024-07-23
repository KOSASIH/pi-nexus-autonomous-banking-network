const network = require('./network');

const api = {
  async getBalance() {
    const response = await network.makeApiRequest('balance', 'get');
    return response.data;
  },

  async getTransactions() {
    const response = await network.makeApiRequest('transactions', 'get');
    return response.data;
  },

  async sendTransaction(data) {
    const response = await network.makeApiRequest('send', 'post', data);
    return response.data;
  },
};

module.exports = api;
