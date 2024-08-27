import axios from 'axios';

class Api {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.axios = axios.create({
      baseURL: apiUrl,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  }

  async get(endpoint) {
    return this.axios.get(endpoint);
  }

  async post(endpoint, data) {
    return this.axios.post(endpoint, data);
  }

  async put(endpoint, data) {
    return this.axios.put(endpoint, data);
  }

  async delete(endpoint) {
    return this.axios.delete(endpoint);
  }
}

export default Api;
