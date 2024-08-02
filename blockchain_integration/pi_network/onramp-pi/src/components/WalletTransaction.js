import React, { useState, useEffect } from 'react';
import { useWeb3React } from '@web3-react/core';
import { ethers } from 'ethers';
import axios from 'axios';
import { useToasts } from 'react-toast-notifications';

const WalletTransaction = () => {
  const { account, library } = useWeb3React();
  const [transaction, setTransaction] = useState({});
  const [loading, setLoading] = useState(false);
  const { addToast } = useToasts();

  useEffect(() => {
    const fetchTransaction = async () => {
      try {
        const response = await axios.get(`/api/transactions/${transaction.id}`);
        setTransaction(response.data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchTransaction();
  }, [transaction.id]);

  const handleConfirmTransaction = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`/api/transactions/${transaction.id}/confirm`);
      addToast('Transaction confirmed successfully!', {
        appearance: 'success',
        autoDismiss: true,
      });
    } catch (error) {
      console.error(error);
      addToast('Failed to confirm transaction!', {
        appearance: 'error',
        autoDismiss: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Transaction Details</h2>
      <p>Transaction ID: {transaction.id}</p>
      <p>Amount: {transaction.amount} ETH</p>
      <p>Status: {transaction.status}</p>
      {transaction.status === 'pending' ? (
        <button onClick={handleConfirmTransaction}>Confirm Transaction</button>
      ) : (
        <p>Transaction already confirmed</p>
      )}
    </div>
  );
};

export default WalletTransaction;
