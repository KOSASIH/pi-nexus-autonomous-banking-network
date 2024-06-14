import React, { useState, useEffect } from 'eact';
import axios from 'axios';
import WebSocket from 'ws';

const TransactionFeed = () => {
  const [transactions, setTransactions] = useState([]);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    axios.get('/api/transactions')
     .then(response => {
        setTransactions(response.data);
      })
     .catch(error => {
        console.error(error);
      });

    const wsUrl = 'ws://localhost:8080/transactions';
    const wsOptions = {
      // You can add headers, query params, etc. here
    };

    setWs(new WebSocket(wsUrl, wsOptions));

    ws.onmessage = (event) => {
      const newTransaction = JSON.parse(event.data);
      setTransactions((prevTransactions) => [...prevTransactions, newTransaction]);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
    };
  }, []);

  return (
    <div>
      <h2>Transaction Feed</h2>
      <ul>
        {transactions.map((transaction) => (
          <li key={transaction.id}>
            {transaction.amount} {transaction.currency} - {transaction.description}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TransactionFeed;
