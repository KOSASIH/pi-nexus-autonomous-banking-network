

const API = axios.create({
  baseURL: "https://api.example.com",
  timeout: 10000,
  headers: {

  },
  // Enable HTTP/2
  http2: true,
  // Enable keep-alive
  keepAlive: true,
});

// Add retry mechanism with exponential backoff
API.interceptors.request.use(
  (config) => {

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
  return API.post('/notifications', { message });
};
