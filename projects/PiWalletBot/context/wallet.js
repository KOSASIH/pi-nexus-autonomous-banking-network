import { useState, useEffect } from 'react';
import { api } from './api';

const WalletContext = React.createContext();

const WalletProvider = ({ children }) => {
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    const address = localStorage.getItem('address');
    if (address) {
      api.getPiBalance(address).then((response) => {
        setBalance(response.data);
      });
      api.getTransactions(address).then((response) => {
        setTransactions(response.data);
      });
    }
  }, []);

  const getBalance = async () => {
    try {
      const response = await api.getPiBalance(localStorage.getItem('address'));
      setBalance(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const getTransactions = async () => {
    try {
      const response = await api.getTransactions(localStorage.getItem('address'));
      setTransactions(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <WalletContext.Provider value={{ balance, transactions, getBalance, getTransactions }}>
      {children}
    </WalletContext.Provider
