import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.sidrachain.com',
});

export function getBalance(address) {
  return api.get(`/balances/${address}`);
}

export function getTransactionHistory(address) {
  return api.get(`/transactions/${address}`);
}
