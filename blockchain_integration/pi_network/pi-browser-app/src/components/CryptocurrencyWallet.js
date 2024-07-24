import React, { useState, useEffect } from 'react';
import { CryptocurrencyAPI } from '../api';

const CryptocurrencyWallet = () => {
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sendAmount, setSendAmount] = useState(0);
  const [sendRecipient, setSendRecipient] = useState('');

  useEffect(() => {
    const fetchBalance = async () => {
      const response = await CryptocurrencyAPI.getBalance();
      setBalance(response.data);
      setLoading(false);
    };
    fetchBalance();
  }, []);

  useEffect(() => {
    const fetchTransactions = async () => {
      const response = await CryptocurrencyAPI.getTransactions();
      setTransactions(response.data);
    };
    fetchTransactions();
  }, []);

  const handleSendTransaction = async (amount, recipient) => {
    setLoading(true);
    const response = await CryptocurrencyAPI.sendTransaction(amount, recipient);
    setBalance(response.data.balance);
    setTransactions(response.data.transactions);
    setLoading(false);
  };

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    if (name === 'sendAmount') {
      setSendAmount(value);
    } else if (name === 'sendRecipient') {
      setSendRecipient(value);
    }
  };

  return (
    <div className="cryptocurrency-wallet">
      <h1>Cryptocurrency Wallet</h1>
      <p>Balance: {balance} Pi Coins</p>
      <ul className="transaction-list">
        {transactions.map((transaction) => (
          <li key={transaction.id}>
            <p>Transaction ID: {transaction.id}</p>
            <p>Amount: {transaction.amount} Pi Coins</p>
            <p>Recipient: {transaction.recipient}</p>
          </li>
        ))}
      </ul>
      <form>
        <label>
          Send Amount:
          <input type="number" name="sendAmount" value={sendAmount} onChange={handleInputChange} />
        </label>
        <label>
          Recipient:
          <input type="text" name="sendRecipient" value={sendRecipient} onChange={handleInputChange} />
        </label>
        <button onClick={() => handleSendTransaction(sendAmount, sendRecipient)}>Send</button>
      </form>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <p>Wallet is ready!</p>
      )}
    </div>
  );
};

export default CryptocurrencyWallet;
