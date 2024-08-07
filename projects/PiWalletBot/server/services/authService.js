import jwt from 'jsonwebtoken';

const authService = {
  async login(username, password) {
    // authenticate user with Pi Network API
    const response = await apiService.post('/login', { username, password });
    const token = response.data.token;
    return token;
  },

  async register(username, password) {
    // register user with Pi Network API
    const response = await apiService.post('/register', { username, password });
    const token = response.data.token;
    return token;
  },

  async verifyToken(token) {
    try {
      const decoded = jwt.verify(token, process.env.SECRET_KEY);
      return decoded;
    } catch (error) {
      return null;
    }
  },
};

export default authService;
