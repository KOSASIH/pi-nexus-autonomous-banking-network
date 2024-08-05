// blockchain_integration/pi_network.js
const axios = require('axios');

class PiNetwork {
  async createAccount(userIdentity, walletAddress) {
    const apiUrl = 'https://api.minepi.com/v1/account';
    const headers = {
      'Content-Type': 'application/json',
    };

    const data = {
      user_identity: userIdentity,
      wallet_address: walletAddress,
    };

    try {
      const response = await axios.post(apiUrl, data, headers);
      const account = response.data;
      return account;
    } catch (error) {
      console.error(error);
      throw error;
    }
  }
}

module.exports = PiNetwork;
