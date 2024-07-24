import React, { useState, useEffect } from 'react';
import * as web3 from 'web3';
import * as ethers from 'ethers';

const PiBrowserBlockchainExplorer = () => {
  const [blockchainNetwork, setBlockchainNetwork] = useState(null);
  const [blockNumber, setBlockNumber] = useState(0);
  const [transactionHistory, setTransactionHistory] = useState([]);
  const [blockchainExplorer, setBlockchainExplorer] = useState(null);

  useEffect(() => {
    // Initialize blockchain network
    const network = new web3.eth.net();
    network.setProvider(new web3.providers.HttpProvider('https://mainnet.infura.io'));
    setBlockchainNetwork(network);

    // Initialize blockchain explorer
    const explorer = new ethers.providers.EtherscanProvider('mainnet');
    setBlockchainExplorer(explorer);
  }, []);

  const handleBlockNumberChange = (blockNumber) => {
    setBlockNumber(blockNumber);
    getTransactionHistory();
  };

  const getTransactionHistory = () => {
    blockchainExplorer.getBlockNumber().then((blockNumber) => {
      blockchainExplorer.getTransactionHistory(blockNumber).then((transactions) => {
        setTransactionHistory(transactions);
      });
    });
  };

  return (
    <div>
      <h1>Blockchain Explorer</h1>
      <section>
        <h2>Block Number</h2>
        <input type="number" value={blockNumber} onChange={(e) => handleBlockNumberChange(e.target.value)} />
      </section>
      <section>
        <h2>Transaction History</h2>
        <ul>
          {transactionHistory.map((transaction) => (
            <li key={transaction.hash}>{transaction.hash}</li>
          ))}
        </ul>
      </section>
    </div>
  );
};
