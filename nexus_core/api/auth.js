import axios from 'axios';

const authenticate = async (username, password) => {
  try {
    const response = await axios.post('/api/auth', { username, password });
    return response.data.token;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export default authenticate;
