import { useState, useEffect } from 'react';
import { useApi } from './useApi';

const useAuth = () => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();

  useEffect(() => {
    const checkToken = async () => {
      try {
        const token = localStorage.getItem('token');
        if (token) {
          const response = await api.get('/auth/validate', { headers: { Authorization: `Bearer ${token}` } });
          if (response.data.isValid) {
            setUser(response.data.user);
            setToken(token);
          }
        }
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };
    checkToken();
  }, []);

  const login = async (username, password) => {
    try {
      setLoading(true);
      const response = await api.post('/auth/login', { username, password });
      setUser(response.data.user);
      setToken(response.data.token);
      localStorage.setItem('token', response.data.token);
    } catch (error) {
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  const register = async (username, password, email) => {
    try {
      setLoading(true);
      const response = await api.post('/auth/register', { username, password, email });
      setUser(response.data.user);
      setToken(response.data.token);
      localStorage.setItem('token', response.data.token);
    } catch (error) {
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
  };

  return { user, token, loading, error, login, register, logout };
};

export default useAuth;
