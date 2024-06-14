import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.pi.network',
});

export const fetchPortfolioData = () => api.get('/portfolio');
