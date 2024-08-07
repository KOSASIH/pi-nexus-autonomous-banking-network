import React from 'react';
import { useAuth } from '../context/auth';
import { useApi } from '../context/api';

const Dashboard = () => {
  const auth = useAuth();
  const api = useApi();

  const handleGetBalance = async () => {
    try {
      const balance = await api.getPiBalance(auth.user.address);
      console.log(`Your balance is ${balance} Pi coins`);
    } catch (error) {
      console.error(error);
    }
  };

  const handleSendTransaction = async () => {
    try {
      const amount = 10; // hardcoded for demo purposes
      const recipient = 'recipient_address'; // hardcoded for demo purposes
      const transaction = await api.sendPiTransaction(auth.user.address, recipient, amount);
      console.log(`Transaction sent: ${transaction}`);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="dashboard">
      <h1>Welcome, {auth.user.username}!</h1>
      <Button onClick={handleGetBalance}>Get Balance</Button>
      <Button onClick={handleSendTransaction}>Send Transaction</Button>
    </div>
  );
};

export default Dashboard;
