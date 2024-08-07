import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.pi.network',
});

export const getPiBalance = async () => {
  const response = await api.get('/balance');
  return response.data;
};

export const sendPiTransaction = async (amount, recipient) => {
  const response = await api.post('/transaction', {
    amount,
    recipient,
  });
  return response.data;
};

export default api;
