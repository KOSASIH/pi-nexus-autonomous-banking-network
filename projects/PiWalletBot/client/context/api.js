import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.pi.network',
});

export const getPiBalance = async (address) => {
  try {
    const response = await api.get(`/balance/${address}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const sendPiTransaction = async (from, to, amount) => {
  try {
    const response = await api.post('/transaction', {
      from,
      to,
      amount,
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
