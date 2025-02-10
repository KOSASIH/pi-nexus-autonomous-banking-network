import axios from 'axios';

const API_URL = 'http://localhost:3000/api'; // Adjust based on your backend URL

export const registerUser  = async (userData) => {
    return await axios.post(`${API_URL}/users`, userData);
};

export const loginUser  = async (userData) => {
    return await axios.post(`${API_URL}/users/login`, userData);
};

export const requestPasswordReset = async (email) => {
    return await axios.post(`${API_URL}/password-reset/request`, { email });
};

export const resetPassword = async (token, newPassword) => {
    return await axios.post(`${API_URL}/password-reset/reset`, { token, newPassword });
};

export const getUser Transactions = async (token) => {
    return await axios.get(`${API_URL}/transactions`, {
        headers: { Authorization: `Bearer ${token}` },
    });
};
