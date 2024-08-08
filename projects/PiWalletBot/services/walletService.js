import axios from 'axios';
import { API_URL } from '../config';

class WalletService {
  async getBalance() {
    try {
      const response = await axios.get(`${API_URL}/wallet/balance`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async sendTransaction(to, value) {
    try {
      const response = await axios.post(`${API_URL}/wallet/send-transaction`, {
        to,
        value,
      });
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async getTransactionHistory() {
    try {
      const response = await axios.get(`${API_URL}/wallet/transaction-history`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }
}

export default new WalletService();
