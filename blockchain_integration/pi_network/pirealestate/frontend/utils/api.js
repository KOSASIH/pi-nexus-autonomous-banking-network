import axios from "axios";
import { API_URL } from "../constants";
import { getToken } from "../storage";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.push({
  request: (config) => {
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  response: (response) => {
    return response;
  },
  error: (error) => {
    if (error.response.status === 401) {
      // Handle unauthorized error
      console.error("Unauthorized access");
    }
    return Promise.reject(error);
  },
});

export const get = (url, params) => api.get(url, { params });
export const post = (url, data) => api.post(url, data);
export const put = (url, data) => api.put(url, data);
export const del = (url) => api.delete(url);

export const setToken = (token) => {
  api.defaults.headers.common.Authorization = `Bearer ${token}`;
};

export const removeToken = () => {
  delete api.defaults.headers.common.Authorization;
};
