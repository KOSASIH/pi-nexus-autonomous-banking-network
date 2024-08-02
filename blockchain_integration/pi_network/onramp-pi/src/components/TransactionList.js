import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useToasts } from 'react-toast-notifications';
import WalletTransaction from './WalletTransaction';

const TransactionList = () => {
  const [transactions, setTransactions] = useState([]);
  const { addToast } = useToasts();

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const response = await axios.get('/api/transactions');
        setTransactions(response.data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchTransactions();
  }, []);

  return (
    <div>
      <h1>Transaction List</h1>
      <ul>
        {transactions.map((transaction) => (
          <li key={transaction.id}>
            <WalletTransaction transaction={transaction} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TransactionList;
