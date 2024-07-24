import React, { useState, useEffect } from 'react';
import { BlockchainAPI } from '../api';

const BlockchainExplorer = () => {
  const [blocks, setBlocks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBlocks = async () => {
      const response = await BlockchainAPI.getBlocks();
      setBlocks(response.data);
      setLoading(false);
    };
    fetchBlocks();
  }, []);

  return (
    <div className="blockchain-explorer">
      <h1>Blockchain Explorer</h1>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <ul className="block-list">
          {blocks.map((block) => (
            <li key={block.hash}>
              <h2>Block {block.height}</h2>
              <p>Hash: {block.hash}</p>
              <p>Transactions: {block.transactions.length}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default BlockchainExplorer;
