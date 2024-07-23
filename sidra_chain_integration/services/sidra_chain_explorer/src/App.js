// sidra_chain_explorer/src/App.js
import React, { useState, useEffect } from 'react';
import { useQuery, gql } from '@apollo/client';
import { ethers } from 'ethers';

const BLOCKCHAIN_EXPLORER_QUERY = gql`
  query BlockchainExplorer($blockNumber: Int) {
    blockchain {
      block(number: $blockNumber) {
        transactions {
          id
          from
          to
          value
        }
      }
    }
  }
`;

function App() {
  const [blockNumber, setBlockNumber] = useState(0);
  const { data, loading, error } = useQuery(BLOCKCHAIN_EXPLORER_QUERY, {
    variables: { blockNumber },
  });

  useEffect(() => {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    provider.getBlockNumber().then((blockNumber) => setBlockNumber(blockNumber));
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h1>Sidra Chain Explorer</h1>
      <p>Block Number: {blockNumber}</p>
      <ul>
        {data.blockchain.block.transactions.map((transaction) => (
          <li key={transaction.id}>
            <span>From: {transaction.from}</span>
            <span>To: {transaction.to}</span>
            <span>Value: {transaction.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
