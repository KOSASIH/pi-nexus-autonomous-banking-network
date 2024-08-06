import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export const createPolicy = (data) => api.post('/policies', data);
export const getPolicies = () => api.get('/policies');
export const getPolicy = (id) => api.get(`/policies/${id}`);
export const updatePolicy = (id, data) => api.put(`/policies/${id}`, data);
export const deletePolicy = (id) => api.delete(`/policies/${id}`);
