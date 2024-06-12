// api/walletAPI.js
import axios from 'axios';

const API_URL = 'https://wallet-api.example.com/api';

export const getAddress = async () => {
  try {
    const response = await axios.get(`${API_URL}/address`);
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const getBalance = async () => {
  try {
    const response = await axios.get(`${API_URL}/balance`);
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const getTransactions = async () => {
  try {
    const response = await axios.get(`${API_URL}/transactions`);
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const sendTransaction = async (amount, recipient) => {
  try {
    const response = await axios.post(`${API_URL}/transactions`, { amount, recipient });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};
