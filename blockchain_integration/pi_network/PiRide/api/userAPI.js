import axios from 'axios';
import { API_URL } from '../constants';

const userAPI = {
  getUserProfile: async (userId) => {
    try {
      const response = await axios.get(`${API_URL}/users/${userId}`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  updateUserProfile: async (userId, userData) => {
    try {
      const response = await axios.patch(`${API_URL}/users/${userId}`, userData);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  getUserRideHistory: async (userId) => {
    try {
      const response = await axios.get(`${API_URL}/users/${userId}/rides`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },
};

export default userAPI;
