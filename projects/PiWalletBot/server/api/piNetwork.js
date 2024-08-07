import axios from 'axios';

const piNetworkApi = axios.create({
  baseURL: 'https://api.pi.network',
});

export const getPiBalance = async (address) => {
  const response = await piNetworkApi.get(`/balance/${address}`);
  return response.data;
};

export const sendPiTransaction = async (from, to, amount) => {
  const response = await piNetworkApi.post('/transaction', {
    from,
    to,
    amount,
  });
  return response.data;
};
