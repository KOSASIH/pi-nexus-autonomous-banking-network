import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.example.com',
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.push({
  request: (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
});

export const getPiBalance = async (address) => {
  try {
    const response = await api.get(`/pi/balance/${address}`);
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const getTransactions = async (address) => {
  try {
    const response = await api.get(`/pi/transactions/${address}`);
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const sendMessage = async (address, message) => {
  try {
    const response = await api.post(`/pi/messages/${address}`, { message });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const login = async (username, password) => {
  try {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const register = async (username, password, email) => {
  try {
    const response = await api.post('/auth/register', { username, password, email });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const validateToken = async (token) => {
  try {
    const response = await api.post('/auth/validate', { token });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};
