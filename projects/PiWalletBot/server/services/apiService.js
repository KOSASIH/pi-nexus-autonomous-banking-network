import axios from 'axios';

const apiService = axios.create({
  baseURL: 'https://api.pi.network',
});

export const getPiBalance = async (address) => {
  const response = await apiService.get(`/balance/${address}`);
  return response.data;
};

export const sendPiTransaction = async (from, to, amount) => {
  const response = await apiService.post('/transaction', {
    from,
    to,
    amount,
  });
  return response.data;
};

export default apiService;
