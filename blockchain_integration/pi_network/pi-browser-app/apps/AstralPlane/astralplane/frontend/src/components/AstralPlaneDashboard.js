import React, { useState, useEffect } from 'react';
import { useWeb3React } from '@web3-react/core';
import { AstralPlaneToken } from '../contracts/AstralPlaneToken';

const AstralPlaneDashboard = () => {
  const { account, library } = useWeb3React();
  const [balance, setBalance] = useState(0);
  const [tokenBalance, setTokenBalance] = useState(0);

  useEffect(() => {
    const fetchBalance = async () => {
      const balance = await library.getBalance(account);
      setBalance(balance);
    };
    fetchBalance();
  }, [account, library]);

  useEffect(() => {
    const fetchTokenBalance = async () => {
      const tokenBalance = await AstralPlaneToken.balanceOf(account);
      setTokenBalance(tokenBalance);
    };
    fetchTokenBalance();
  }, [account, AstralPlaneToken]);

  return (
    <div>
      <h1>AstralPlane Dashboard</h1>
      <p>Account: {account}</p>
      <p>Balance: {balance} ETH</p>
      <p>Token Balance: {tokenBalance} APT</p>
    </div>
  );
};

export default AstralPlaneDashboard;
