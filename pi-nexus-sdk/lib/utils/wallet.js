import { Api } from './api';

class Wallet {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.api = new Api(apiUrl, apiKey);
  }

  async getWallets() {
    return this.api.get('/v1/wallets');
  }

  async getWallet(walletId) {
    return this.api.get(`/v1/wallets/${walletId}`);
  }

  async createWallet(walletData) {
    return this.api.post('/v1/wallets', walletData);
  }
}

export default Wallet;
