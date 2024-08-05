import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.ai-platform.com',
  headers: {
    'Content-Type': 'application/json'
  }
});

api.interceptors.push({
  request: (config) => {
    // Add authentication token to request headers
    config.headers.Authorization = `Bearer ${localStorage.getItem('token')}`;
    return config;
  },
  response: (response) => {
    // Cache response data for 1 hour
    const cacheKey = `api-${response.config.url}`;
    localStorage.setItem(cacheKey, JSON.stringify(response.data));
    localStorage.setItem(`${cacheKey}-expires`, Date.now() + 3600000);
    return response;
  },
  error: (error) => {
    // Handle API errors
    console.error(error);
    return Promise.reject(error);
  }
});

export default api;
