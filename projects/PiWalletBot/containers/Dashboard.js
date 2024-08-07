import React, { useState, useEffect } from 'react';
import { useApi } from '../context/api';
import { useAuth } from '../context/auth';
import DashboardComponent from '../components/Dashboard';

const Dashboard = () => {
  const api = useApi();
  const auth = useAuth();
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const getBalance = async () => {
      try {
        const response = await api.getPiBalance(auth.user.address);
        setBalance(response.data);
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    const getTransactions = async () => {
      try {
        const response = await api.getTransactions(auth.user.address);
        setTransactions(response.data);
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    getBalance();
    getTransactions();
  }, []);

  if (loading) {
    return <Loading />;
  }

  return (
    <DashboardComponent
      balance={balance}
      transactions={transactions}
      onGetBalance={getBalance}
      onGetTransactions={getTransactions}
    />
  );
};

export default Dashboard;
