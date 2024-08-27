import React from 'react';
import { Link } from 'react-router-dom';
import { Transaction } from '../types';

interface Props {
  transaction: Transaction;
}

const TransactionComponent: React.FC<Props> = ({ transaction }) => {
  return (
    <div>
      <h2>Transaction {transaction.hash}</h2>
      <p>From: {transaction.from}</p>
      <p>To: {transaction.to}</p>
      <p>Value: {transaction.value} Pi</p>
      <Link to={`/contracts/${transaction.contractAddress}`}>View Contract</Link>
    </div>
  );
};

export default TransactionComponent;
