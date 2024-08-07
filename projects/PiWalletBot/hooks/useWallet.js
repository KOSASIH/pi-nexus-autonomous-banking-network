import { useState, useEffect } from 'react';
import { useApi } from './useApi';
import { useAuth } from './useAuth';

const useWallet = () => {
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();
  const { user, token } = useAuth();

  useEffect(() => {
    const getBalance = async () => {
      try {
        setLoading(true);
        const response = await api.get('/pi/balance', { headers: { Authorization: `Bearer ${token}` } });
        setBalance(response.data);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };
    if (user) {
      getBalance();
    }
  }, [user, token]);

  useEffect(() => {
    const getTransactions = async () => {
      try {
        setLoading(true);
        const response = await api.get('/pi/transactions', { headers: { Authorization: `Bearer ${token}` } });
        setTransactions(response.data);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      } finally {
        setLoading(false);
      }
    };
    if (user) {
      getTransactions();
    }
  }, [user, token]);

  const sendTransaction = async (recipient, amount) => {
    try {
      setLoading(true);
      const response = await api.post('/pi/transactions', { recipient, amount }, { headers: { Authorization: `Bearer ${token}` } });
      setTransactions([...transactions, response.data]);
    } catch (error) {
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  return { balance, transactions, loading, error, getBalance, getTransactions, sendTransaction };
};

export default useWallet;
