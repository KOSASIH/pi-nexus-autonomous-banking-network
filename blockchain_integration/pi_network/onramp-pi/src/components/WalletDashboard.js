import React, { useState, useEffect } from 'react';
import { useWeb3React } from '@web3-react/core';
import { ethers } from 'ethers';
import axios from 'axios';
import { useToasts } from 'react-toast-notifications';

const WalletDashboard = () => {
  const { account, library } = useWeb3React();
  const [walletBalance, setWalletBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [onrampUrl, setOnrampUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const { addToast } = useToasts();

  useEffect(() => {
    const fetchWalletBalance = async () => {
      try {
        const balance = await library.getBalance(account);
        setWalletBalance(ethers.utils.formatEther(balance));
      } catch (error) {
        console.error(error);
      }
    };
    fetchWalletBalance();
  }, [account, library]);

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const response = await axios.get('/api/transactions');
        setTransactions(response.data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchTransactions();
  }, []);

  const handleFundWallet = async () => {
    try {
      setLoading(true);
      const response = await axios.post('/api/onramp', {
        address: account,
        email: 'user@example.com',
        redirectUrl: 'https://example.com/redirect',
      });
      setOnrampUrl(response.data.url);
      addToast('On-ramp URL generated successfully!', {
        appearance: 'success',
        autoDismiss: true,
      });
    } catch (error) {
      console.error(error);
      addToast('Failed to generate on-ramp URL!', {
        appearance: 'error',
        autoDismiss: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Wallet Dashboard</h1>
      <p>Wallet Balance: {walletBalance} ETH</p>
      <button onClick={handleFundWallet}>Fund Wallet</button>
      {onrampUrl ? (
        <a href={onrampUrl} target="_blank" rel="noopener noreferrer">
          Click here to complete the on-ramp flow
        </a>
      ) : (
        <p>No on-ramp URL generated</p>
      )}
      <h2>Transactions</h2>
      <ul>
        {transactions.map((transaction) => (
          <li key={transaction.id}>
            {transaction.amount} ETH - {transaction.status}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default WalletDashboard;
