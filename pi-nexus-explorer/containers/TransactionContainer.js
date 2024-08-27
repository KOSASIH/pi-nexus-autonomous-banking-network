import React, { useState, useEffect } from 'react';
import { Transaction } from '../types';
import { apiUtils } from '../utils';
import TransactionComponent from '../components/TransactionComponent';

interface Props {
  match: {
    params: {
      transactionHash: string;
    };
  };
}

const TransactionContainer: React.FC<Props> = ({ match }) => {
  const [transaction, setTransaction] = useState<Transaction | null>(null);

  useEffect(() => {
    apiUtils.getTransaction(match.params.transactionHash).then((transaction) => setTransaction(transaction));
  }, [match.params.transactionHash]);

  return (
    <div>
      {transaction ? <TransactionComponent transaction={transaction} /> : <p>Loading...</p>}
    </div>
  );
};

export default TransactionContainer;
