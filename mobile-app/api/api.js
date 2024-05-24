import axios from "axios";
import { retry, throttle, debounce } from "lodash";
import { v4 as uuidv4 } from "uuid";
import { encrypt, decrypt } from "./crypto";
import { logger } from "./logger";

const API = axios.create({
  baseURL: "https://api.example.com",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
    "X-Request-ID": uuidv4(),
  },
  // Enable HTTP/2
  http2: true,
  // Enable keep-alive
  keepAlive: true,
});

// Add retry mechanism with exponential backoff
API.interceptors.request.use(
  (config) => {
    config.retry = retry(3, 1000, (err) => {
      logger.error(`Request failed with error: ${err}`);
    });
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add throttle mechanism to prevent excessive requests
API.interceptors.request.use(
  (config) => {
    config.throttle = throttle(5, 1000, () => {
      logger.warn("Request throttled");
    });
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add debounce mechanism to prevent duplicate requests
API.interceptors.request.use(
  (config) => {
    config.debounce = debounce(500, () => {
      logger.info("Request debounced");
    });
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add encryption and decryption mechanisms
API.interceptors.request.use(
  (config) => {
    if (config.data) {
      config.data = encrypt(config.data);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

API.interceptors.response.use(
  (response) => {
    if (response.data) {
      response.data = decrypt(response.data);
    }
    return response;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add authentication mechanism
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
    if (response.status === 401) {
      localStorage.removeItem("token");
      // Redirect to login screen
    }
    return response;
  },
  (error) => {
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

export const sendNotification = (message) => {
  return API.post("/notifications", { message });
};
