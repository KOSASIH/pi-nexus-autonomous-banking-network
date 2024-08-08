import axios from 'axios';
import { API_URL } from '../config';
import { localStorage } from '../utils';

class AuthService {
  async register(username, email, password) {
    try {
      const response = await axios.post(`${API_URL}/auth/register`, {
        username,
        email,
        password,
      });
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async login(email, password) {
    try {
      const response = await axios.post(`${API_URL}/auth/login`, {
        email,
        password,
      });
      const token = response.data.token;
      localStorage.setItem('token', token);
      return token;
    } catch (error) {
      throw error.response.data;
    }
  }

  async forgotPassword(email) {
    try {
      const response = await axios.post(`${API_URL}/auth/forgot-password`, {
        email,
      });
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async resetPassword(token, password) {
    try {
      const response = await axios.post(`${API_URL}/auth/reset-password`, {
        token,
        password,
      });
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }

  async logout() {
    localStorage.removeItem('token');
  }

  async isLoggedIn() {
    return !!localStorage.getItem('token');
  }
}

export default new AuthService();
