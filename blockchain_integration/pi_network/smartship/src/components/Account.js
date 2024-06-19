import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import { useContractLoader, useContractReader } from 'eth-hooks';

const Account = ({ address, provider, blockNumber }) => {
  const [balance, setBalance] = useState(0);
  const contract = useContractLoader('MyContract', provider);

  useEffect(() => {
    async function fetchData() {
      const balance = await contract.methods.balanceOf(address).call();
      setBalance(Web3.utils.fromWei(balance, 'ether'));
    }
    fetchData();
  }, [address, blockNumber]);

  return (
    <div>
      <h2>Account: {address}</h2>
      <p>Balance: {balance} ETH</p>
    </div>
  );
};

export default Account;
