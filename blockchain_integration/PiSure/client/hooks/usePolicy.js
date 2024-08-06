import { useState, useEffect } from 'react';
import axios from 'axios';

const usePolicy = () => {
  const [policies, setPolicies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPolicies = async () => {
      setLoading(true);
      try {
        const response = await axios.get('/api/policies');
        setPolicies(response.data);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, []);

  return { policies, loading, error };
};

export default usePolicy;
