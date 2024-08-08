import axios from 'axios';
import { API_URL } from '../config';

class ApiService {
  async makeRequest(method, endpoint, data = {}) {
    try {
      const response = await axios({
        method,
        url: `${API_URL}${endpoint}`,
        data,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async get(endpoint) {
    return this.makeRequest('GET', endpoint);
  }

  async post(endpoint, data) {
    return this.makeRequest('POST', endpoint, data);
  }

  async put(endpoint, data) {
    return this.makeRequest('PUT', endpoint, data);
  }

  async delete(endpoint) {
    return this.makeRequest('DELETE', endpoint);
  }
}

export default new ApiService();
