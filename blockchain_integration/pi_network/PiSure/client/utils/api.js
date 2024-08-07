import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export const createPolicy = (data) => api.post('/policies', data);
export const getPolicies = () => api
