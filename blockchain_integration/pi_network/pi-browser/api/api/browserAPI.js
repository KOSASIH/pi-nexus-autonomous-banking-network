// api/browserAPI.js
import axios from 'axios';

const API_URL = 'https://browser-api.example.com/api';

export const compileCode = async (htmlCode, cssCode, jsCode) => {
  try {
    const response = await axios.post(`${API_URL}/compile`, { htmlCode, cssCode, jsCode });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const saveCode = async (htmlCode, cssCode, jsCode) => {
  try {
    const response = await axios.post(`${API_URL}/save`, { htmlCode, cssCode, jsCode });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};
