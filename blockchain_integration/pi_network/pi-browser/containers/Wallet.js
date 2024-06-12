// containers/Wallet.js
import React from 'react';
import { useSelector } from 'react-redux';
import { selectWalletAddress, selectWalletBalance } from '../reducers/walletReducer';

const Wallet = () => {
  const walletAddress = useSelector(selectWalletAddress);
  const walletBalance = useSelector(selectWalletBalance);

  return (
    <div className="wallet">
      <h1>Wallet</h1>
      <p>Address: {walletAddress}</p>
      <p>Balance: {walletBalance}</p>
    </div>
  );
};

export default Wallet;
