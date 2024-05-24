import axios from "axios";

const API = axios.create({
  baseURL: "https://api.example.com",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

API.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response.status === 401) {
      localStorage.removeItem("token");
      // Redirect to login screen
    }
    return Promise.reject(error);
  },
);

export const authenticate = (username, password) => {
  return API.post("/authenticate", { username, password });
};

export const getAccounts = () => {
  return API.get("/accounts");
};

export const getTransactions = () => {
  return API.get("/transactions");
};
