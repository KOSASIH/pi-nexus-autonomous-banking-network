import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [accountNumber, setAccountNumber] = useState('');
  const [amount, setAmount] = useState(0);
  const [transactionStatus, setTransactionStatus] = useState('');

  useEffect(() => {
    axios.post('/transaction', { amount, account_number: accountNumber })
      .then(response => {
        setTransactionStatus(response.data.message);
      })
      .catch(error => {
        setTransactionStatus('Error processing transaction');
      });
  }, [amount, accountNumber]);

  return (
    <div>
      <h1>Nexus Bank</h1>
      <form>
        <label>Account Number:</label>
        <input type="text" value={accountNumber} onChange={e => setAccountNumber(e.target.value)} />
        <br />
        <label>Amount:</label>
        <input type="number" value={amount} onChange={e => setAmount(e.target.value)} />
        <br />
        <button type="submit">Process Transaction</button>
      </form>
      <p>{transactionStatus}</p>
    </div>
  );
}

export default App;
