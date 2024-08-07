import { useState, useEffect } from 'react';
import axios from 'axios';

const useApi = () => {
  const [api, setApi] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const createApi = async () => {
      try {
        const response = await axios.get('/api/config');
        const apiConfig = response.data;
        const apiInstance = axios.create({
          baseURL: apiConfig.baseURL,
          headers: {
            'Content-Type': 'application/json',
          },
        });
        setApi(apiInstance);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };
    createApi();
  }, []);

  const get = async (endpoint, params) => {
    try {
      setLoading(true);
      const response = await api.get(endpoint, { params });
      return response.data;
    } catch (error) {
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const post = async (endpoint, data) => {
    try {
      setLoading(true);
      const response = await api.post(endpoint, data);
      return response.data;
    } catch (error) {
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { api, loading, error, get, post };
};

export default useApi;
