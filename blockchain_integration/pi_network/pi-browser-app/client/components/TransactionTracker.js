import React from 'react';
import { useSelector } from 'react-redux';
import { getTransactionDetails } from '../actions';

const TransactionTracker = ({ transactions }) => {
  const transactionDetails = useSelector((state) => state.transactionDetails);

  return (
    <div>
      <h2>Transaction Tracker</h2>
      <ul>
        {transactions.map((transaction) => (
          <li key={transaction.id}>
            <span>{transaction.amount}</span>
            <button onClick={() => getTransactionDetails(transaction.id)}>
              View Details
            </button>
          </li>
        ))}
      </ul>
      {transactionDetails && (
        <div>
          <h3>Transaction Details</h3>
          <p>ID: {transactionDetails.id}</p>
          <p>Amount: {transactionDetails.amount}</p>
          <p>Status: {transactionDetails.status}</p>
        </div>
      )}
    </div>
  );
};

export default TransactionTracker;
