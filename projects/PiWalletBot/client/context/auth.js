import jwt from 'jsonwebtoken';

const auth = {
  async login(username, password) {
    try {
      const response = await api.post('/login', { username, password });
      const token = response.data.token;
      localStorage.setItem('token', token);
      return token;
    } catch (error) {
      throw error;
    }
  },

  async register(username, password) {
    try {
      const response = await api.post('/register', { username, password });
      const token = response.data.token;
      localStorage.setItem('token', token);
      return token;
    } catch (error) {
      throw error;
    }
  },

  async verifyToken(token) {
    try {
      const decoded = jwt.verify(token, process.env.SECRET_KEY);
      return decoded;
    } catch (error) {
      return null;
    }
  },

  async logout() {
    localStorage.removeItem('token');
  },
};

export default auth;
