import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { getTransactions } from '../actions';
import TransactionTracker from './TransactionTracker';

const Dashboard = () => {
  const dispatch = useDispatch();
  const transactions = useSelector((state) => state.transactions);
  const [showTransactionTracker, setShowTransactionTracker] = useState(false);

  useEffect(() => {
    dispatch(getTransactions());
  }, []);

  const handleShowTransactionTracker = () => {
    setShowTransactionTracker(true);
  };

  return (
    <div>
      <h1>Dashboard</h1>
      <button onClick={handleShowTransactionTracker}>Track Transactions</button>
      {showTransactionTracker && <TransactionTracker transactions={transactions} />}
    </div>
  );
};

export default Dashboard;
