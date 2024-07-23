const axios = require('axios');
const { api_url, api_key, api_secret } = require('./config.json');

const network = {
  async getApiUrl() {
    return api_url;
  },

  async getApiKey() {
    return api_key;
  },

  async getApiSecret() {
    return api_secret;
  },

  async makeApiRequest(endpoint, method, data) {
    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${api_key}`,
    };

    try {
      const response = await axios[method](`${api_url}/${endpoint}`, data, {
        headers,
      });
      return response.data;
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  },
};

module.exports = network;
