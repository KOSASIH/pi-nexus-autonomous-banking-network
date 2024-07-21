// backend/sidraChain.js
const axios = require('axios');

const sidraChain = {
  async verifyIdentity(userAddress, userData) {
    try {
      // Replace with your Sidra Chain API endpoint
      const response = await axios.post('https://sidra-chain-api.com/verify', {
        userAddress,
        userData,
      });
      return response.data;
    } catch (error) {
      console.error(`Error verifying user on Sidra Chain: ${error}`);
      throw error;
    }
  },
};

module.exports = sidraChain;
