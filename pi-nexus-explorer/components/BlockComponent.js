import React from 'react';
import { Link } from 'react-router-dom';
import { Block } from '../types';

interface Props {
  block: Block;
}

const BlockComponent: React.FC<Props> = ({ block }) => {
  return (
    <div>
      <h2>Block {block.number}</h2>
      <p>Hash: {block.hash}</p>
      <p>Transactions: {block.transactions.length}</p>
      <Link to={`/transactions?block=${block.number}`}>View Transactions</Link>
    </div>
  );
};

export default BlockComponent;
